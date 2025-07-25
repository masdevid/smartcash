"""
File: smartcash/model/training/data_loader_factory.py
Deskripsi: Factory untuk membuat data loaders dari preprocessed dataset dengan YOLO format
"""

import os
import torch
import cv2
import numpy as np
import atexit
import weakref
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
import yaml


class YOLODataset(Dataset):
    """Dataset class untuk YOLO format dengan preprocessed .npy files"""
    
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get preprocessed .npy files (pre_**.npy dan aug_**_{variance}.npy)
        npy_files = list(self.images_dir.glob('*.npy'))
        self.image_files = [f for f in npy_files if f.name.startswith(('pre_', 'aug_'))]
        self.image_files.sort()
        
        # Filter files dengan label yang ada
        self.valid_files = []
        for npy_file in self.image_files:
            label_file = self.labels_dir / f"{npy_file.stem}.txt"
            if label_file.exists():
                self.valid_files.append(npy_file)
    
    def __len__(self) -> int:
        return len(self.valid_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npy_path = self.valid_files[idx]
        label_path = self.labels_dir / f"{npy_path.stem}.txt"
        
        # Load preprocessed .npy file (already normalized float32)
        image = np.load(npy_path)  # Shape: (H, W, C) atau (C, H, W)
        
        # Ensure correct format (C, H, W) and float32
        if len(image.shape) == 3:
            if image.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                image = image.transpose(2, 0, 1)
            # Already in (C, H, W) format
        
        # Load labels
        labels = self._load_labels(label_path)
        
        # Convert ke tensor (data sudah normalized dari preprocessor)
        image = torch.from_numpy(image.astype(np.float32))
        if len(labels) > 0:
            # Keep class IDs as integers and coordinates as floats
            labels_tensor = torch.from_numpy(labels.astype(np.float32))
            # Ensure class column (first column) is integer type
            labels_tensor[:, 0] = labels_tensor[:, 0].long().float()
            labels = labels_tensor
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, labels
    
    def _load_labels(self, label_path: Path) -> np.ndarray:
        """Load YOLO format labels"""
        if not label_path.exists():
            return np.zeros((0, 5))
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    labels.append([cls_id, x, y, w, h])
        
        return np.array(labels) if labels else np.zeros((0, 5))
    
    def _resize_image_and_labels(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Data sudah di-resize dari preprocessor, skip resize"""
        # Preprocessed data sudah dalam format yang benar dari preprocessor
        # Labels sudah dalam format normalized, tidak perlu adjust
        return image, labels

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function untuk YOLO batch"""
    images, labels = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Combine labels dengan batch index
    targets = []
    for i, label in enumerate(labels):
        if len(label) > 0:
            # Add batch index as first column (ensure float for consistency)
            batch_labels = torch.full((len(label), 1), float(i), dtype=torch.float32)
            targets.append(torch.cat([batch_labels, label], 1))
    
    targets = torch.cat(targets, 0) if targets else torch.zeros((0, 6), dtype=torch.float32)
    
    return images, targets

class DataLoaderFactory:
    """Factory untuk membuat data loaders dari preprocessed dataset"""
    _instances = weakref.WeakSet()
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, data_dir: str = 'data/preprocessed'):
        self.config = config or self._load_default_config()
        self.data_dir = Path(data_dir)
        self._validate_data_structure()
        self._dataloaders = []
        DataLoaderFactory._instances.add(self)
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """Clean up all resources used by dataloaders and their workers to prevent semaphore leaks"""
        for dl in self._dataloaders:
            try:
                # Explicitly shutdown worker processes if possible
                if hasattr(dl, '_shutdown_workers'):
                    dl._shutdown_workers()
                if hasattr(dl, '_iterator'):
                    # Remove iterator to help GC
                    delattr(dl, '_iterator')
                if hasattr(dl, 'dataset') and hasattr(dl.dataset, 'close'):
                    dl.dataset.close()
            except Exception:
                pass
        self._dataloaders.clear()
        if self in DataLoaderFactory._instances:
            DataLoaderFactory._instances.remove(self)
    
    @classmethod
    def cleanup_all(cls):
        """Clean up all DataLoaderFactory instances"""
        for instance in list(cls._instances):
            instance.cleanup()
    
    def __del__(self):
        self.cleanup()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default training config"""
        config_path = Path('smartcash/configs/training_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback config jika file tidak ada"""
        return {
            'training': {
                'batch_size': 16,
                'data': {
                    'num_workers': 4,  # Use optimal workers for mixed I/O and CPU tasks
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 2,
                    'drop_last': True
                }
            }
        }
    
    def _validate_data_structure(self) -> None:
        """Validate struktur data preprocessed"""
        required_splits = ['train', 'valid']
        for split in required_splits:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if not images_dir.exists():
                raise FileNotFoundError(f"❌ Images directory tidak ditemukan: {images_dir}")
            if not labels_dir.exists():
                raise FileNotFoundError(f"❌ Labels directory tidak ditemukan: {labels_dir}")
    
    def create_train_loader(self, img_size: int = 640) -> DataLoader:
        """Create training data loader"""
        images_dir = self.data_dir / 'train' / 'images'
        labels_dir = self.data_dir / 'train' / 'labels'
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=img_size,
            augment=True
        )
        
        data_config = self.config.get('training', {}).get('data', {})
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 16),
            shuffle=True,
            num_workers=4,
            pin_memory=data_config.get('pin_memory', True),
            persistent_workers=data_config.get('persistent_workers', True),
            prefetch_factor=data_config.get('prefetch_factor', 2),
            drop_last=data_config.get('drop_last', True),
            collate_fn=collate_fn
        )
        self._dataloaders.append(loader)
        return loader
    
    def create_val_loader(self, img_size: int = 640) -> DataLoader:
        """Create validation data loader"""
        images_dir = self.data_dir / 'valid' / 'images'
        labels_dir = self.data_dir / 'valid' / 'labels'
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=img_size,
            augment=False
        )
        
        data_config = self.config.get('training', {}).get('data', {})
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 16),
            shuffle=False,
            num_workers=4,
            pin_memory=data_config.get('pin_memory', True),
            persistent_workers=data_config.get('persistent_workers', True),
            collate_fn=collate_fn
        )
        self._dataloaders.append(loader)
        return loader
    
    def create_test_loader(self, img_size: int = 640) -> Optional[DataLoader]:
        """Create test data loader jika tersedia"""
        images_dir = self.data_dir / 'test' / 'images'
        labels_dir = self.data_dir / 'test' / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            return None
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=img_size,
            augment=False
        )
        
        data_config = self.config.get('training', {}).get('data', {})
        loader = DataLoader(
            dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 16),
            shuffle=False,
            num_workers=4,
            pin_memory=data_config.get('pin_memory', True),
            collate_fn=collate_fn
        )
        self._dataloaders.append(loader)
        return loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get informasi dataset untuk preprocessed files"""
        info = {}
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                # Count .npy files (pre_*.npy dan aug_*.npy)
                npy_files = list(images_dir.glob('*.npy'))
                preprocessed_files = [f for f in npy_files if f.name.startswith(('pre_', 'aug_'))]
                
                info[split] = {
                    'num_images': len(preprocessed_files),
                    'preprocessed_files': len([f for f in preprocessed_files if f.name.startswith('pre_')]),
                    'augmented_files': len([f for f in preprocessed_files if f.name.startswith('aug_')]),
                    'images_dir': str(images_dir),
                    'labels_dir': str(labels_dir)
                }
            else:
                info[split] = {'num_images': 0, 'available': False}
        
        return info
    
    def get_class_distribution(self, split: str = 'train') -> Dict[int, int]:
        """Get distribusi kelas untuk split tertentu"""
        labels_dir = self.data_dir / split / 'labels'
        
        if not labels_dir.exists():
            return {}
        
        class_counts = {}
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        
        return class_counts

# Convenience functions
def create_data_loaders(config: Optional[Dict] = None, data_dir: str = 'data/preprocessed', 
                       img_size: int = 640) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """One-liner untuk create train, val, test loaders"""
    factory = DataLoaderFactory(config, data_dir)
    return factory.create_train_loader(img_size), factory.create_val_loader(img_size), factory.create_test_loader(img_size)

def get_dataset_stats(data_dir: str = 'data/preprocessed') -> Dict[str, Any]:
    """One-liner untuk get dataset statistics"""
    factory = DataLoaderFactory(data_dir=data_dir)
    return factory.get_dataset_info()
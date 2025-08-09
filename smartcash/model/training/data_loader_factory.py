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
    
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640, 
                 augment: bool = False, max_samples: Optional[int] = None):
        """
        Initialize YOLO dataset.
        
        Args:
            images_dir: Directory containing image .npy files
            labels_dir: Directory containing label .txt files
            img_size: Target image size (height=width)
            augment: Whether to apply data augmentation
            max_samples: Maximum number of samples to use (for testing)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get preprocessed .npy files (pre_**.npy and aug_**_{variance}.npy)
        npy_files = list(self.images_dir.glob('*.npy'))
        self.image_files = [f for f in npy_files if f.name.startswith(('pre_', 'aug_'))]
        self.image_files.sort()
        
        # Filter files with existing labels
        self.valid_files = []
        for npy_file in self.image_files:
            label_file = self.labels_dir / f"{npy_file.stem}.txt"
            if label_file.exists():
                self.valid_files.append(npy_file)
                
        # Limit number of samples if max_samples is specified
        self.max_samples = max_samples
        if self.max_samples is not None and self.max_samples > 0:
            self.valid_files = self.valid_files[:self.max_samples]
            print(f"[DEBUG] YOLODataset: Limited to {self.max_samples} samples")
    
    def __len__(self) -> int:
        length = len(self.valid_files)
        if hasattr(self, 'max_samples') and self.max_samples is not None:
            print(f"[DEBUG] YOLODataset.__len__: Returning {length} samples (max_samples={self.max_samples})")
        return length
    
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
                # For PyTorch 2.7+, use proper iterator cleanup
                if hasattr(dl, '_iterator') and dl._iterator is not None:
                    # Safely shutdown the iterator which handles worker processes
                    if hasattr(dl._iterator, '_shutdown_workers'):
                        dl._iterator._shutdown_workers()
                    elif hasattr(dl._iterator, 'shutdown'):
                        dl._iterator.shutdown()
                    # Remove iterator reference to help GC
                    dl._iterator = None
                    
                # Fallback for older PyTorch versions
                if hasattr(dl, '_shutdown_workers'):
                    dl._shutdown_workers()
                    
                # Clean up dataset resources
                if hasattr(dl, 'dataset') and hasattr(dl.dataset, 'close'):
                    dl.dataset.close()
                    
            except (AttributeError, RuntimeError) as e:
                # Ignore AttributeError for missing attributes in different PyTorch versions
                # Ignore RuntimeError for workers already shutdown
                pass
            except Exception:
                # Catch any other unexpected exceptions during cleanup
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
        """Fallback config jika file tidak ada."""
        import torch
        
        # Check if running on Apple Silicon (MPS)
        is_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        if is_mps:
            # Optimized settings for Apple Silicon (MPS)
            return {
                'training': {
                    'batch_size': 16,
                    'data': {
                        'num_workers': 0,  # Disable multiprocessing for MPS
                        'pin_memory': False,  # Not needed for MPS
                        'persistent_workers': False,  # Disable for MPS
                        'prefetch_factor': 2,
                        'drop_last': True,
                        'multiprocessing_context': 'forkserver',
                        'timeout': 30
                    }
                }
            }
        
        # Default settings for non-MPS devices
        pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        
        # Determine optimal number of workers based on CPU count
        try:
            cpu_count = os.cpu_count() or 1
            # Set optimal workers to a safe value (e.g., half of CPU cores, max 8)
            optimal_workers = min(cpu_count // 2 if cpu_count > 1 else 1, 8)
        except NotImplementedError:
            optimal_workers = 2  # Fallback for systems where cpu_count is not available

        # For PyTorch 2.7+, reduce multiprocessing complexity to avoid worker issues
        if pytorch_version >= (2, 7):
            num_workers = min(optimal_workers, 4)  # Limit to 4 for PyTorch 2.7+ compatibility
            persistent_workers = False  # Disable persistent workers to avoid _workers_status issues
        else:
            num_workers = optimal_workers
            persistent_workers = True
            
        return {
            'training': {
                'batch_size': 16,
                'data': {
                    'num_workers': num_workers,
                    'pin_memory': True,
                    'persistent_workers': persistent_workers,
                    'prefetch_factor': 2,
                    'drop_last': True,
                    'timeout': 30  # Add timeout to prevent hanging workers
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
    
    def create_train_loader(self, img_size: int = 640, max_samples: Optional[int] = None) -> DataLoader:
        """
        Create training data loader
        
        Args:
            img_size: Target image size (height=width)
            max_samples: Maximum number of samples to use (for testing)
            
        Returns:
            DataLoader for training data
        """
        print(f"[DEBUG] create_train_loader called with max_samples={max_samples}")
        images_dir = self.data_dir / 'train' / 'images'
        labels_dir = self.data_dir / 'train' / 'labels'
        print(f"[DEBUG] Loading training data from {images_dir}")
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=img_size,
            augment=True,
            max_samples=max_samples
        )
        
        # Get platform-specific configuration
        from smartcash.model.training.platform_presets import PlatformPresets
        presets = PlatformPresets()
        platform_config = presets.get_data_config()
        
        # Get config with platform overrides
        data_config = {**platform_config, **self.config.get('training', {}).get('data', {})}
        
        # Get batch size with proper precedence: training.batch_size > training.data.batch_size > platform
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size')
        if batch_size is None:
            # Check data config within training config
            batch_size = training_config.get('data', {}).get('batch_size')
        if batch_size is None:
            # Fall back to platform config
            batch_size = platform_config.get('batch_size', 16)
        
        # Determine source for logging
        if 'batch_size' in training_config:
            batch_size_source = "training.batch_size"
        elif 'batch_size' in training_config.get('data', {}):
            batch_size_source = "training.data.batch_size"
        else:
            batch_size_source = "platform"
        
        print(f"📊 Training DataLoader Configuration:")
        print(f"   • Batch Size: {batch_size} (source: {batch_size_source})")
        print(f"   • Dataset Size: {len(dataset)} samples")
        print(f"   • Batches per Epoch: {len(dataset) // batch_size}")
        
        # Determine multiprocessing settings with PyTorch version compatibility
        num_workers = data_config.get('num_workers', 8)
        persistent_workers = data_config.get('persistent_workers', True)
        
        # If persistent_workers is enabled but num_workers is 0, disable persistent_workers
        if num_workers == 0:
            persistent_workers = False
            
        print(f"   • Workers: {num_workers}, Pin Memory: {data_config.get('pin_memory', True)}")
        print(f"   • Prefetch Factor: {data_config.get('prefetch_factor', 2)}, Drop Last: {data_config.get('drop_last', True)}")
        print(f"   • Persistent Workers: {persistent_workers}, Non-blocking: {data_config.get('non_blocking', False)}")
        
        # Log performance mode if maximum speed settings detected
        if num_workers >= 8 and data_config.get('prefetch_factor', 2) >= 4:
            print("⚡ MAXIMUM SPEED MODE: High-performance dataloader configuration detected")
            
        loader_args = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': data_config.get('pin_memory', True),
            'prefetch_factor': data_config.get('prefetch_factor', 2),
            'drop_last': data_config.get('drop_last', True),
            'collate_fn': collate_fn
        }
        
        # Add persistent_workers only if num_workers > 0
        if num_workers > 0:
            loader_args['persistent_workers'] = persistent_workers
            
        # Add timeout if specified
        if data_config.get('timeout'):
            loader_args['timeout'] = data_config.get('timeout')
            
        loader = DataLoader(**loader_args)
        
        # Add non_blocking support for faster tensor operations (used in training loop)
        # Note: non_blocking is not a DataLoader parameter, so we attach it after creation
        if data_config.get('non_blocking', False):
            loader._non_blocking = True  # Store for training loop access
        self._dataloaders.append(loader)
        return loader
    
    def create_val_loader(self, img_size: int = 640, max_samples: Optional[int] = None) -> DataLoader:
        print(f"[DEBUG] create_val_loader called with max_samples={max_samples}")
        """
        Create validation data loader
        
        Args:
            img_size: Target image size (height=width)
            max_samples: Maximum number of samples to use (for testing)
            
        Returns:
            DataLoader for validation data
        """
        images_dir = self.data_dir / 'valid' / 'images'
        labels_dir = self.data_dir / 'valid' / 'labels'
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=img_size,
            augment=False,
            max_samples=max_samples
        )
        
        # Get platform-specific configuration
        from smartcash.model.training.platform_presets import PlatformPresets
        presets = PlatformPresets()
        platform_config = presets.get_data_config()
        
        # Get config with platform overrides
        data_config = {**platform_config, **self.config.get('training', {}).get('data', {})}
        
        # Get batch size with proper precedence: training.batch_size > training.data.batch_size > platform
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size')
        if batch_size is None:
            # Check data config within training config
            batch_size = training_config.get('data', {}).get('batch_size')
        if batch_size is None:
            # Fall back to platform config
            batch_size = platform_config.get('batch_size', 16)
        
        # Determine source for logging
        if 'batch_size' in training_config:
            batch_size_source = "training.batch_size"
        elif 'batch_size' in training_config.get('data', {}):
            batch_size_source = "training.data.batch_size"
        else:
            batch_size_source = "platform"
        
        print(f"📊 Validation DataLoader Configuration:")
        print(f"   • Batch Size: {batch_size} (source: {batch_size_source})")
        print(f"   • Dataset Size: {len(dataset)} samples")
        print(f"   • Batches per Epoch: {len(dataset) // batch_size}")
        
        # Determine multiprocessing settings with PyTorch version compatibility
        num_workers = data_config.get('num_workers', 4)
        persistent_workers = data_config.get('persistent_workers', True)
        
        # If persistent_workers is enabled but num_workers is 0, disable persistent_workers
        if num_workers == 0:
            persistent_workers = False
            
        loader_args = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': data_config.get('pin_memory', True),
            'collate_fn': collate_fn
        }
        
        # Add persistent_workers only if num_workers > 0
        if num_workers > 0:
            loader_args['persistent_workers'] = persistent_workers
            
        # Add timeout if specified
        if data_config.get('timeout'):
            loader_args['timeout'] = data_config.get('timeout')
            
        loader = DataLoader(**loader_args)
        self._dataloaders.append(loader)
        return loader
    
    def create_test_loader(self, img_size: int = 640, max_samples: Optional[int] = None) -> Optional[DataLoader]:
        print(f"[DEBUG] create_test_loader called with max_samples={max_samples}")
        """
        Create test data loader if available
        
        Args:
            img_size: Target image size (height=width)
            max_samples: Maximum number of samples to use (for testing)
            
        Returns:
            DataLoader for test data if available, else None
        """
        images_dir = self.data_dir / 'test' / 'images'
        labels_dir = self.data_dir / 'test' / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            return None
        
        dataset = YOLODataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            img_size=img_size,
            augment=False,
            max_samples=max_samples
        )
        
        data_config = self.config.get('training', {}).get('data', {})
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        
        # Handle auto batch size detection
        if batch_size is None:
            # Import platform presets for auto-detection
            from smartcash.model.training.platform_presets import PlatformPresets
            presets = PlatformPresets()
            
            # Get recommended batch size for current platform
            data_config = presets.get_data_config()
            batch_size = data_config.get('batch_size', 16)
            print(f"🤖 Auto-detected test batch size: {batch_size} for platform: {presets.platform_info.get('platform_name', 'unknown')}")
        
        # Determine multiprocessing settings with PyTorch version compatibility
        num_workers = data_config.get('num_workers', 4)
        persistent_workers = data_config.get('persistent_workers', True)
        
        # If persistent_workers is enabled but num_workers is 0, disable persistent_workers
        if num_workers == 0:
            persistent_workers = False
            
        loader_args = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': data_config.get('pin_memory', True),
            'collate_fn': collate_fn
        }
        
        # Add persistent_workers only if num_workers > 0
        if num_workers > 0:
            loader_args['persistent_workers'] = persistent_workers
            
        # Add timeout if specified
        if data_config.get('timeout'):
            loader_args['timeout'] = data_config.get('timeout')
            
        loader = DataLoader(**loader_args)
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
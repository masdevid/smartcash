"""
File: smartcash/dataset/services/loader/preprocessed_dataset_loader.py
Deskripsi: Fixed loader untuk dataset preprocessed dengan reduced duplication dan SRP compliance
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory


class PreprocessedDatasetLoader:
    """Loader untuk dataset preprocessed dengan service layer integration."""
    
    def __init__(
        self,
        preprocessed_dir: Union[str, Path] = "data/preprocessed",
        fallback_to_raw: bool = True,
        auto_preprocess: bool = True,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        self.logger = logger or get_logger()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.fallback_to_raw = fallback_to_raw
        self.auto_preprocess = auto_preprocess
        
        # Default config
        self.config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'img_size': [640, 640],
                'normalize': True,
                'output_dir': str(self.preprocessed_dir),
                'num_workers': 4
            },
            'batch_size': 16,
            'pin_memory': True
        }
        
        if config:
            if 'preprocessing' in config:
                self.config['preprocessing'].update(config['preprocessing'])
            if 'data' in config:
                self.config['data'].update(config['data'])
            for key in ['batch_size', 'pin_memory']:
                if key in config:
                    self.config[key] = config[key]
        
        self._preprocessing_manager = None
        self.logger.info(f"📦 PreprocessedDatasetLoader initialized (dir: {self.preprocessed_dir})")
    
    def get_dataset(
        self,
        split: str,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        """Get dataset untuk split dengan fallback mechanism."""
        normalized_split = 'valid' if split == 'val' else split
        
        if not self._is_split_available(normalized_split):
            if self.auto_preprocess:
                self.logger.info(f"🔄 Auto preprocessing {normalized_split}")
                self._ensure_preprocessing(normalized_split)
            elif self.fallback_to_raw:
                self.logger.warning(f"⚠️ Using raw dataset for {normalized_split}")
                return self._get_raw_dataset(normalized_split, require_all_layers, transform)
            else:
                raise ValueError(f"Preprocessed dataset unavailable: {normalized_split}")
        
        return PreprocessedDataset(
            root_dir=self.preprocessed_dir / normalized_split,
            img_size=tuple(self.config['preprocessing']['img_size']),
            require_all_layers=require_all_layers,
            transform=transform,
            logger=self.logger
        )
    
    def get_dataloader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        """Get dataloader untuk split."""
        # Use defaults
        batch_size = kwargs.get('batch_size', self.config['batch_size'])
        num_workers = kwargs.get('num_workers', self.config['preprocessing']['num_workers'])
        shuffle = kwargs.get('shuffle', split in ['train', 'training'])
        
        dataset = self.get_dataset(
            split=split,
            require_all_layers=kwargs.get('require_all_layers', False),
            transform=kwargs.get('transform')
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config['pin_memory'],
            drop_last=(split in ['train', 'training'])
        )
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, torch.utils.data.DataLoader]:
        """Get dataloader untuk semua split."""
        dataloaders = {}
        for split in ['train', 'valid', 'test']:
            try:
                dataloaders[split] = self.get_dataloader(split=split, **kwargs)
            except Exception as e:
                self.logger.warning(f"⚠️ Skip dataloader {split}: {str(e)}")
        return dataloaders
    
    def ensure_preprocessed(self, splits: List[str] = ['train', 'valid', 'test'], 
                          force_reprocess: bool = False) -> Dict[str, Dict[str, Any]]:
        """Ensure dataset preprocessed menggunakan service layer."""
        results = {}
        for split in splits:
            try:
                if force_reprocess or not self._is_split_available(split):
                    result = self._ensure_preprocessing(split, force_reprocess)
                    results[split] = result
                else:
                    self.logger.info(f"✅ Dataset preprocessed {split} available")
                    results[split] = {'success': True, 'message': 'Available'}
            except Exception as e:
                self.logger.error(f"❌ Error preprocessing {split}: {str(e)}")
                results[split] = {'success': False, 'error': str(e)}
        return results
    
    def _ensure_preprocessing(self, split: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Ensure preprocessing using service layer."""
        try:
            if not self._preprocessing_manager:
                self._preprocessing_manager = PreprocessingFactory.create_preprocessing_manager(self.config, self.logger)
            
            result = self._preprocessing_manager.coordinate_preprocessing(split=split, force_reprocess=force_reprocess)
            
            if result['success']:
                self.logger.success(f"✅ Preprocessing {split}: {result.get('total_images', 0)} gambar")
            else:
                self.logger.error(f"❌ Preprocessing {split} failed: {result.get('message')}")
            
            return result
        except Exception as e:
            self.logger.error(f"❌ Preprocessing error {split}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _is_split_available(self, split: str) -> bool:
        """Check split availability."""
        split_dir = self.preprocessed_dir / split
        if not split_dir.exists():
            return False
        
        images_dir = split_dir / 'images'
        if not images_dir.exists():
            return False
        
        # Check for any image files
        for ext in ['.jpg', '.jpeg', '.png', '.npy']:
            if list(images_dir.glob(f'*{ext}')):
                return True
        return False
    
    def _get_raw_dataset(self, split: str, require_all_layers: bool = False, 
                        transform: Optional[Callable] = None) -> torch.utils.data.Dataset:
        """Fallback to raw dataset."""
        try:
            from smartcash.dataset.services.loader.multilayer_loader import MultilayerDataset
            raw_dir = Path(self.config['data']['dir']) / split
            return MultilayerDataset(
                root_dir=raw_dir,
                img_size=tuple(self.config['preprocessing']['img_size']),
                require_all_layers=require_all_layers,
                transform=transform
            )
        except ImportError:
            return PreprocessedDataset(
                root_dir=Path(self.config['data']['dir']) / split,
                img_size=tuple(self.config['preprocessing']['img_size']),
                require_all_layers=require_all_layers,
                transform=transform,
                logger=self.logger
            )


class PreprocessedDataset(torch.utils.data.Dataset):
    """Dataset untuk preprocessed data dengan multilayer support."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        img_size: Tuple[int, int] = (640, 640),
        require_all_layers: bool = False,
        transform: Optional[Callable] = None,
        logger: Optional[Any] = None
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.require_all_layers = require_all_layers
        self.transform = transform
        self.logger = logger or get_logger()
        
        self.images_dir = self.root_dir / 'images'
        self.labels_dir = self.root_dir / 'labels'
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        if not self.labels_dir.exists():
            self.logger.warning(f"⚠️ Labels directory not found: {self.labels_dir}")
            self.labels_dir = None
        
        # Get image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.npy']:
            self.image_files.extend(sorted(self.images_dir.glob(f'*{ext}')))
        
        # Filter valid items
        self.valid_items = []
        for img_path in self.image_files:
            if self.labels_dir:
                label_path = self.labels_dir / img_path.with_suffix('.txt').name
                if label_path.exists():
                    self.valid_items.append((img_path, label_path))
                elif not self.require_all_layers:
                    self.valid_items.append((img_path, None))
            else:
                self.valid_items.append((img_path, None))
        
        if not self.valid_items:
            raise ValueError(f"No valid items found in {root_dir}")
        
        self.logger.info(f"📦 Loaded {len(self.valid_items)} items from {root_dir}")
    
    def __len__(self) -> int:
        return len(self.valid_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label_path = self.valid_items[idx]
        
        img_tensor = self._load_image(img_path)
        targets = self._load_label(label_path) if label_path else torch.zeros((0, 5))
        
        if self.transform:
            img_tensor, targets = self.transform(img_tensor, targets)
        
        return {
            'image': img_tensor,
            'targets': targets,
            'img_path': str(img_path),
            'label_path': str(label_path) if label_path else None
        }
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load image dengan multiple format support."""
        try:
            if img_path.suffix == '.npy':
                img = np.load(str(img_path))
                img_tensor = torch.from_numpy(img).float()
                if len(img_tensor.shape) == 3 and img_tensor.shape[-1] == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Cannot load: {img_path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            
            return img_tensor
        except Exception as e:
            self.logger.error(f"❌ Error loading {img_path}: {str(e)}")
            return torch.zeros((3, *self.img_size))
    
    def _load_label(self, label_path: Path) -> torch.Tensor:
        """Load YOLO format labels."""
        try:
            if not label_path.exists():
                return torch.zeros((0, 5))
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            targets = []
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5:
                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:5])
                    targets.append([class_id, x, y, w, h])
            
            return torch.tensor(targets, dtype=torch.float32)
        except Exception as e:
            self.logger.error(f"❌ Error loading label {label_path}: {str(e)}")
            return torch.zeros((0, 5))
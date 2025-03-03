# File: smartcash/handlers/data_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk pemrosesan dataset multi-layer

import os
import torch
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger

class MultilayerDataset(Dataset):
    """Dataset untuk deteksi multi-layer uang kertas Rupiah."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path ke direktori dataset
            img_size: Ukuran gambar yang diinginkan
            mode: Mode dataset ('train', 'val', 'test')
            transform: Custom transformations
            logger: Logger opsional
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.logger = logger or SmartCashLogger(__name__)
        
        # Layer configuration berdasarkan struktur class yang aktual
        self.layer_config = {
            'banknote': {
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'class_ids': list(range(7))  # 0-6
            },
            'nominal': {
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'class_ids': list(range(7, 14))  # 7-13
            },
            'security': {
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'class_ids': list(range(14, 17))  # 14-16
            }
        }
        
        # Load images and labels
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
            
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Validate
        if len(self.image_files) == 0:
            self.logger.warning(f"⚠️ No images found in {self.images_dir}")
            
        # Setup transformations
        self.default_transform = A.Compose([
            A.Resize(height=img_size[1], width=img_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        if mode == 'train':
            # Add augmentations for training
            self.default_transform = A.Compose([
                A.Resize(height=img_size[1], width=img_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        self.logger.info(f"📚 Loaded {len(self.image_files)} images for {mode} dataset")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        try:
            img_path = self.image_files[idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load labels
            bboxes = []
            class_labels = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class_id, x, y, w, h
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            else:
                transformed = self.default_transform(image=img, bboxes=bboxes, class_labels=class_labels)
                
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
            # Convert to tensor format
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            
            # Prepare multi-layer targets
            total_classes = sum(len(layer['classes']) for layer in self.layer_config.values())
            targets = torch.zeros((total_classes, 5))  # [class_id, x, y, w, h]
            
            for i, (bbox, cls_id) in enumerate(zip(bboxes, class_labels)):
                x_center, y_center, width, height = bbox
                targets[cls_id, 0] = x_center
                targets[cls_id, 1] = y_center
                targets[cls_id, 2] = width
                targets[cls_id, 3] = height
                targets[cls_id, 4] = 1.0  # Confidence
                
            return img_tensor, targets
            
        except Exception as e:
            self.logger.error(f"❌ Error loading item {idx}: {str(e)}")
            # Return a zero image and empty target as fallback
            img_tensor = torch.zeros((3, self.img_size[1], self.img_size[0]))
            total_classes = sum(len(layer['classes']) for layer in self.layer_config.values())
            targets = torch.zeros((total_classes, 5))
            return img_tensor, targets
            
class DataHandler:
    """Handler untuk mengelola dataset dan dataloader untuk training dan evaluasi."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
    def setup_dataset_structure(self) -> None:
        """Setup struktur folder dataset jika belum ada."""
        data_dir = Path(self.config.get('data_dir', 'data'))
        
        for split in ['train', 'valid', 'test']:
            # Buat direktori images dan labels untuk setiap split
            for subdir in ['images', 'labels']:
                (data_dir / split / subdir).mkdir(parents=True, exist_ok=True)
                
        self.logger.success(f"✅ Struktur dataset berhasil dibuat di {data_dir}")
        
    def get_train_loader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """Get DataLoader untuk training data."""
        # Use values from config if not provided
        batch_size = batch_size or self.config['model']['batch_size']
        num_workers = num_workers or self.config['model']['workers']
        
        # Get path
        train_path = Path(self.config['data']['local']['train'])
        
        # Create dataset
        train_dataset = MultilayerDataset(
            data_path=train_path,
            img_size=tuple(self.config['model']['img_size']),
            mode='train',
            logger=self.logger
        )
        
        # Create dataloader
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """Get DataLoader untuk validation data."""
        # Use values from config if not provided
        batch_size = batch_size or self.config['model']['batch_size']
        num_workers = num_workers or self.config['model']['workers']
        
        # Get path
        val_path = Path(self.config['data']['local']['val'])
        
        # Create dataset
        val_dataset = MultilayerDataset(
            data_path=val_path,
            img_size=tuple(self.config['model']['img_size']),
            mode='val',
            logger=self.logger
        )
        
        # Create dataloader
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_test_loader(
        self,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """Get DataLoader untuk test data."""
        # Use values from config if not provided
        batch_size = batch_size or self.config['model']['batch_size']
        num_workers = num_workers or self.config['model']['workers']
        
        # Get path
        if dataset_path is None:
            dataset_path = self.config['data']['local']['test']
        
        # Create dataset
        test_dataset = MultilayerDataset(
            data_path=dataset_path,
            img_size=tuple(self.config['model']['img_size']),
            mode='test',
            logger=self.logger
        )
        
        # Create dataloader
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
# File: handlers/data_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk dataset management dan augmentasi data lokal

import os
from typing import Dict, Optional, Tuple, List
import yaml
import shutil
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from smartcash.utils.logger import SmartCashLogger

class SmartCashDataset(Dataset):
    """Dataset untuk training dan evaluasi SmartCash."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        conditions: Optional[str] = None,
        transform=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path ke direktori dataset
            img_size: Ukuran gambar (width, height)
            mode: Mode dataset ('train', 'val', 'test', 'eval')
            conditions: Kondisi pengujian ('position', 'lighting', None)
            transform: Optional transformasi tambahan
            config: Optional konfigurasi tambahan
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.conditions = conditions
        self.transform = transform
        self.config = config or {}
        
        # Load image files
        self.image_files = sorted(list(self.data_path.glob('*.jpg')))
        self.label_files = [
            self.data_path.parent / 'labels' / f"{img.stem}.txt"
            for img in self.image_files
        ]
        
        # Validate files
        self._validate_files()
    
    def _validate_files(self) -> None:
        """Validate image and label files."""
        if not self.image_files:
            raise FileNotFoundError(
                f"Tidak ada file gambar (.jpg) di {self.data_path}"
            )
        
        missing_labels = [
            label for label in self.label_files
            if not label.exists()
        ]
        if missing_labels:
            raise FileNotFoundError(
                f"Label tidak ditemukan untuk: {[l.stem for l in missing_labels]}"
            )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply conditions
        if self.conditions == 'lighting':
            img = self._adjust_lighting(img)
        elif self.conditions == 'position':
            img = self._adjust_position(img)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        
        # Load and process label
        label = self._load_label(self.label_files[idx])
        
        # Apply transforms
        if self.transform:
            img, label = self.transform(img, label)
        
        return img, label
    
    def _adjust_lighting(self, img: np.ndarray) -> np.ndarray:
        """Adjust image lighting based on config."""
        if 'gamma' in self.config:
            gamma = self.config['gamma']
            img = np.power(img / 255.0, gamma) * 255.0
            img = img.astype(np.uint8)
        return img
    
    def _adjust_position(self, img: np.ndarray) -> np.ndarray:
        """Adjust image position based on config."""
        if 'rotation' in self.config:
            angle = self.config['rotation']
            center = (img.shape[1] // 2, img.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        return img
    
    def _load_label(self, label_path: Path) -> torch.Tensor:
        """Load and process label file."""
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                labels.append(values)
        return torch.tensor(labels, dtype=torch.float32)


class DataHandler:
    """Handler untuk mengelola dataset lokal."""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str = "data",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize handler.
        
        Args:
            config_path: Path ke file konfigurasi
            data_dir: Path ke direktori data
            logger: Optional logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load dataset configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['dataset']
    
    def setup_dataset_structure(self) -> None:
        """Setup dataset folder structure."""
        self.logger.start("Menyiapkan struktur folder dataset...")
        
        try:
            # Create main directory
            self.data_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            for split in ['train', 'valid', 'test']:
                split_dir = self.data_dir / split
                split_dir.mkdir(exist_ok=True)
                
                # Create images and labels subdirectories
                (split_dir / 'images').mkdir(exist_ok=True)
                (split_dir / 'labels').mkdir(exist_ok=True)
            
            self.logger.success("Struktur folder dataset berhasil disiapkan ✨")
            
        except Exception as e:
            self.logger.error(f"Gagal menyiapkan struktur folder: {str(e)}")
            raise
    
    def get_train_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Get training data loader."""
        dataset = SmartCashDataset(
            data_path=self.data_dir / 'train' / 'images',
            mode='train',
            config=self.config
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    
    def get_val_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Get validation data loader."""
        dataset = SmartCashDataset(
            data_path=self.data_dir / 'valid' / 'images',
            mode='val',
            config=self.config
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    
    def get_test_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        conditions: Optional[str] = None,
        **kwargs
    ) -> DataLoader:
        """Get test data loader."""
        dataset = SmartCashDataset(
            data_path=self.data_dir / 'test' / 'images',
            mode='test',
            conditions=conditions,
            config=self.config
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    
    def get_eval_loader(
        self,
        dataset_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        conditions: Optional[str] = None,
        **kwargs
    ) -> DataLoader:
        """Get evaluation data loader for specific dataset."""
        dataset = SmartCashDataset(
            data_path=dataset_path,
            mode='eval',
            conditions=conditions,
            config=self.config
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
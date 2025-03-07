# File: smartcash/handlers/data_manager.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk manajemen dataset dengan dukungan cache yang diperbarui dan konfigurasi layer terpusat

import os
import torch
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import random
import yaml
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.enhanced_cache import EnhancedCache  # Gunakan enhanced cache
from smartcash.utils.layer_config_manager import get_layer_config  # Gunakan layer config manager

class MultilayerDataset(Dataset):
    """Dataset untuk deteksi multilayer dengan konfigurasi layer terpusat."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        layers: Optional[List[str]] = None,
        require_all_layers: bool = False,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi MultilayerDataset.
        
        Args:
            data_path: Path ke direktori dataset
            img_size: Ukuran target gambar
            mode: Mode dataset ('train', 'val', 'test')
            transform: Transformasi kustom
            layers: Daftar layer yang akan diaktifkan
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            logger: Logger kustom
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.require_all_layers = require_all_layers
        self.logger = logger or SmartCashLogger("multilayer_dataset")
        
        # Dapatkan konfigurasi layer dari layer config manager
        self.layer_config_manager = get_layer_config()
        self.layers = layers or self.layer_config_manager.get_layer_names()
        
        # Setup jalur direktori
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        # Validasi direktori
        if not self.images_dir.exists():
            self.logger.warning(f"⚠️ Direktori gambar tidak ditemukan: {self.images_dir}")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
        if not self.labels_dir.exists():
            self.logger.warning(f"⚠️ Direktori label tidak ditemukan: {self.labels_dir}")
            self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Cari file gambar dan validasi label
        self.valid_samples = self._validate_dataset()
        
        if len(self.valid_samples) == 0:
            self.logger.warning(f"⚠️ Tidak ada sampel valid yang ditemukan di {self.data_path}")
    
    def _validate_dataset(self) -> List[Dict]:
        """
        Validasi dataset untuk memastikan file gambar dan label valid.
        
        Returns:
            List info sampel valid
        """
        valid_samples = []
        
        # Cari semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(self.images_dir.glob(ext)))
        
        self.logger.info(f"🔍 Memvalidasi dataset: {len(image_files)} gambar ditemukan")
        
        # Progress bar untuk validasi
        for img_path in tqdm(image_files, desc="Validasi Dataset"):
            sample_info = {
                'image_path': img_path,
                'label_path': self.labels_dir / f"{img_path.stem}.txt",
                'available_layers': [],
                'is_valid': False
            }
            
            # Validasi gambar
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                sample_info['image_size'] = (img.shape[1], img.shape[0])  # (width, height)
            except Exception:
                continue
            
            # Validasi label
            if sample_info['label_path'].exists():
                try:
                    # Parse label untuk menentukan layer yang tersedia
                    layer_classes = self._parse_label_file(sample_info['label_path'])
                    
                    # Periksa layer yang tersedia
                    for layer in self.layers:
                        layer_config = self.layer_config_manager.get_layer_config(layer)
                        class_ids = set(layer_config['class_ids'])
                        if any(cls_id in class_ids for cls_id in layer_classes):
                            sample_info['available_layers'].append(layer)
                    
                    # Tentukan validitas sampel
                    if self.require_all_layers:
                        # Harus memiliki semua layer yang diminta
                        required_layers = set(self.layers)
                        available_layers = set(sample_info['available_layers'])
                        sample_info['is_valid'] = required_layers.issubset(available_layers)
                    else:
                        # Harus memiliki setidaknya satu layer yang diminta
                        sample_info['is_valid'] = any(layer in self.layers for layer in sample_info['available_layers'])
                        
                except Exception:
                    sample_info['is_valid'] = False
            
            if sample_info['is_valid']:
                valid_samples.append(sample_info)
        
        # Log hasil validasi
        self.logger.info(f"✅ Dataset tervalidasi: {len(valid_samples)}/{len(image_files)} sampel valid")
        if self.require_all_layers:
            self.logger.info(f"ℹ️ Mode require_all_layers=True: Hanya menerima sampel dengan semua layer: {self.layers}")
        else:
            self.logger.info(f"ℹ️ Mode require_all_layers=False: Menerima sampel dengan minimal 1 layer dari: {self.layers}")
            
        # Log distribusi layer
        layer_counts = {layer: 0 for layer in self.layer_config_manager.get_layer_names()}
        for sample in valid_samples:
            for layer in sample['available_layers']:
                layer_counts[layer] += 1
                
        for layer, count in layer_counts.items():
            if layer in self.layers and count > 0:
                self.logger.info(f"📊 Layer '{layer}': {count} sampel")
            
        return valid_samples
    
    def _parse_label_file(self, label_path: Path) -> List[int]:
        """
        Parse file label untuk mendapatkan class ID.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List class ID
        """
        class_ids = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # Format YOLO: class_id x y w h
                    try:
                        cls_id = int(float(parts[0]))
                        class_ids.append(cls_id)
                    except ValueError:
                        continue
        
        return class_ids
    
    def __len__(self) -> int:
        """Mendapatkan jumlah sampel valid."""
        return len(self.valid_samples)
    
    def _normalize_bbox(self, bbox: List[float]) -> List[float]:
        """
        Normalisasi koordinat bounding box untuk memastikan berada dalam range [0, 1].
        
        Args:
            bbox: Koordinat bounding box [x_center, y_center, width, height]
            
        Returns:
            Koordinat ternormalisasi
        """
        # Pastikan x_center dan y_center ada dalam [0, 1]
        x_center = max(0.0, min(1.0, bbox[0]))
        y_center = max(0.0, min(1.0, bbox[1]))
        
        # Pastikan width dan height valid
        width = max(0.01, min(1.0, bbox[2]))
        if x_center + width/2 > 1.0:
            width = 2 * (1.0 - x_center)
            
        height = max(0.01, min(1.0, bbox[3]))
        if y_center + height/2 > 1.0:
            height = 2 * (1.0 - y_center)
            
        return [x_center, y_center, width, height]
    
    def _parse_label_by_layer(self, label_path: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
        """
        Parse file label dan kelompokkan berdasarkan layer.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            Dict label per layer
        """
        layer_labels = {layer: [] for layer in self.layers}
        
        # Class ID to layer mapping
        class_to_layer = {}
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            for cls_id in layer_config['class_ids']:
                class_to_layer[cls_id] = layer
        
        # Parse label
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            coords = [float(x) for x in parts[1:5]]
                            
                            # Check if class ID belongs to an active layer
                            if cls_id in class_to_layer and class_to_layer[cls_id] in self.layers:
                                layer = class_to_layer[cls_id]
                                normalized_coords = self._normalize_bbox(coords)
                                layer_labels[layer].append((cls_id, normalized_coords))
                        except (ValueError, IndexError):
                            continue
        
        return layer_labels
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Mendapatkan item dataset dengan format multilayer.
        
        Args:
            idx: Indeks sampel
            
        Returns:
            Dict berisi gambar dan label per layer
        """
        if idx >= len(self.valid_samples):
            raise IndexError(f"Indeks {idx} di luar batas dataset (ukuran: {len(self.valid_samples)})")
            
        sample = self.valid_samples[idx]
        
        # Load gambar
        try:
            img = cv2.imread(str(sample['image_path']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {sample['image_path']}: {str(e)}")
            # Return dummy image sebagai fallback
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # Parse label per layer
        layer_labels = self._parse_label_by_layer(sample['label_path'])
        
        # Siapkan data untuk augmentasi
        all_bboxes = []
        all_classes = []
        for layer in self.layers:
            for cls_id, bbox in layer_labels[layer]:
                all_bboxes.append(bbox)
                all_classes.append(cls_id)
        
        # Augmentasi
        if self.transform:
            try:
                transformed = self.transform(
                    image=img,
                    bboxes=all_bboxes,
                    class_labels=all_classes
                )
                img = transformed['image']
                
                # Reorganisasi hasil augmentasi ke format per layer
                augmented_layer_labels = {layer: [] for layer in self.layers}
                
                # Class ID to layer mapping
                class_to_layer = {}
                for layer in self.layers:
                    layer_config = self.layer_config_manager.get_layer_config(layer)
                    for cls_id in layer_config['class_ids']:
                        class_to_layer[cls_id] = layer
                
                for i, (cls_id, bbox) in enumerate(zip(transformed['class_labels'], transformed['bboxes'])):
                    if cls_id in class_to_layer:
                        layer = class_to_layer[cls_id]
                        augmented_layer_labels[layer].append((cls_id, bbox))
                
                layer_labels = augmented_layer_labels
                
            except Exception as e:
                self.logger.warning(f"⚠️ Augmentasi gagal untuk {sample['image_path']}: {str(e)}")
        else:
            # Resize gambar jika tidak ada augmentasi
            img = cv2.resize(img, self.img_size)
        
        # Konversi gambar ke tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Buat tensor target untuk setiap layer
        targets = {}
        for layer in self.layers:
            if layer in sample['available_layers']:
                # Layer ini tersedia untuk gambar ini
                layer_config = self.layer_config_manager.get_layer_config(layer)
                num_classes = len(layer_config['classes'])
                local_class_ids = layer_config['class_ids']
                
                # Inisialisasi target untuk layer ini
                layer_target = torch.zeros((num_classes, 5))  # [class_id, x, y, w, h]
                
                # Isi dengan data label
                for cls_id, bbox in layer_labels[layer]:
                    # Konversi kembali class ID global ke local dalam layer
                    local_idx = local_class_ids.index(cls_id)
                    x_center, y_center, width, height = bbox
                    layer_target[local_idx, 0] = x_center
                    layer_target[local_idx, 1] = y_center
                    layer_target[local_idx, 2] = width
                    layer_target[local_idx, 3] = height
                    layer_target[local_idx, 4] = 1.0  # Confidence
                
                targets[layer] = layer_target
            else:
                # Layer tidak tersedia, buat tensor kosong
                layer_config = self.layer_config_manager.get_layer_config(layer)
                num_classes = len(layer_config['classes'])
                targets[layer] = torch.zeros((num_classes, 5))
        
        # Tambahkan informasi tambahan untuk debugging
        metadata = {
            'image_path': str(sample['image_path']),
            'label_path': str(sample['label_path']),
            'available_layers': sample['available_layers']
        }
        
        # Return hasil
        return {
            'image': img_tensor,
            'targets': targets,
            'metadata': metadata
        }

class DataManager:
    """
    Unified data management class for SmartCash dengan support untuk EnhancedCache dan LayerConfigManager.
    """
    
    def __init__(
        self,
        config_path: str,
        data_dir: Optional[str] = None,
        cache_size_gb: float = 1.0,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize the DataManager.
        
        Args:
            config_path: Path to configuration file
            data_dir: Base directory for dataset (optional)
            cache_size_gb: Size of preprocessing cache in GB
            logger: Custom logger instance
        """
        self.logger = logger or get_logger("data_manager")
        self.config = self._load_config(config_path)
        
        # Setup paths
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.target_size = tuple(self.config['model']['img_size'])
        
        # Initialize enhanced cache
        self.cache = EnhancedCache(
            max_size_gb=cache_size_gb,
            logger=self.logger
        )
        
        # Initialize layer configuration manager
        self.layer_config = get_layer_config()
        self.active_layers = self.config.get('layers', ['banknote'])
        
        # Setup augmentation pipelines
        self._setup_augmentation_pipelines()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_augmentation_pipelines(self):
        """Setup data augmentation pipelines."""
        # Base transformations
        self.base_transform = A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),
            A.SafeRotate(limit=30, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
    def get_dataloader(
        self, 
        split: str = 'train',
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: Optional[bool] = None,
        require_all_layers: bool = False  # Add this parameter
    ) -> DataLoader:
        """
        Get DataLoader for specific dataset split.
        
        Args:
            split: Dataset split ('train', 'valid', 'test')
            batch_size: Batch size
            num_workers: Number of workers
            shuffle: Whether to shuffle data (defaults to True for train, False otherwise)
            require_all_layers: Whether to require samples to have all active layers
            
        Returns:
            DataLoader instance
        """
        # Determine path from config
        if split == 'train':
            data_path = self.config.get('data', {}).get('local', {}).get('train', 'data/train')
            transform = self.train_transform
            shuffle = True if shuffle is None else shuffle
        elif split == 'valid' or split == 'val':
            data_path = self.config.get('data', {}).get('local', {}).get('valid', 'data/valid')
            transform = self.base_transform
            shuffle = False if shuffle is None else shuffle
        elif split == 'test':
            data_path = self.config.get('data', {}).get('local', {}).get('test', 'data/test')
            transform = self.base_transform
            shuffle = False if shuffle is None else shuffle
        else:
            raise ValueError(f"Invalid split: {split}")
            
        # Create dataset with updated parameters
        dataset = MultilayerDataset(
            data_path=data_path,
            img_size=self.target_size,
            mode=split,
            transform=transform,
            layers=self.active_layers,
            require_all_layers=require_all_layers,  # Pass the new parameter
            logger=self.logger
        )
        
        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._multilayer_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def _multilayer_collate_fn(self, batch):
        """Collate function for enhanced multilayer dataset."""
        # Filter out None values
        batch = [b for b in batch if b is not None and isinstance(b, dict) and 'image' in b and 'targets' in b]
        
        if len(batch) == 0:
            # Return empty batch with layer structure
            dummy_targets = {layer: torch.zeros((0, len(self.layer_config.get_layer_config(layer)['classes']), 5))
                            for layer in self.active_layers}
            return torch.zeros((0, 3, self.target_size[0], self.target_size[1])), dummy_targets
            
        # Extract images and targets
        imgs = [item['image'] for item in batch]
        targets_list = [item['targets'] for item in batch]
        
        # Stack images
        imgs = torch.stack(imgs)
        
        # Combine targets by layer
        combined_targets = {}
        
        for layer in self.active_layers:
            layer_targets = []
            for targets in targets_list:
                if layer in targets:
                    layer_targets.append(targets[layer])
            
            if layer_targets:
                combined_targets[layer] = torch.stack(layer_targets)
            else:
                # If layer missing in some samples, create empty tensor
                num_classes = len(self.layer_config.get_layer_config(layer)['classes'])
                combined_targets[layer] = torch.zeros((len(batch), num_classes, 5))
        
        # Optionally collect metadata if needed
        metadata = [item.get('metadata', {}) for item in batch]
        
        return imgs, combined_targets
        
    def get_train_loader(self, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """Convenience method to get training dataloader."""
        return self.get_dataloader('train', batch_size, num_workers, shuffle=True)
        
    def get_val_loader(self, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """Convenience method to get validation dataloader."""
        return self.get_dataloader('valid', batch_size, num_workers, shuffle=False)
        
    def get_test_loader(self, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """Convenience method to get test dataloader."""
        return self.get_dataloader('test', batch_size, num_workers, shuffle=False)
        
    def get_dataset_stats(self, split: str = 'train') -> Dict:
        """
        Get dataset statistics for a specific split.
        
        Args:
            split: Dataset split ('train', 'valid', 'test')
            
        Returns:
            Dataset statistics
        """
        # Create a temporary dataset to use its validation features
        if split == 'train':
            data_path = self.config.get('data', {}).get('local', {}).get('train', 'data/train')
        elif split == 'valid' or split == 'val':
            data_path = self.config.get('data', {}).get('local', {}).get('valid', 'data/valid')
        elif split == 'test':
            data_path = self.config.get('data', {}).get('local', {}).get('test', 'data/test')
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Create dataset without transform for validation only
        temp_dataset = MultilayerDataset(
            data_path=data_path,
            img_size=self.target_size,
            mode=split,
            transform=None,
            layers=self.active_layers,
            logger=self.logger
        )
        
        # Get basic statistics
        stats = {
            'image_count': len(temp_dataset.image_files),
            'label_count': len([p for p in temp_dataset.valid_samples if p['label_path'].exists()]),
            'layer_stats': {layer: 0 for layer in self.active_layers},
            'class_stats': {},
            'original': 0,
            'augmented': 0
        }
        
        # Analyze layer and class distributions
        for sample in temp_dataset.valid_samples:
            # Count samples by layer
            for layer in sample['available_layers']:
                if layer in stats['layer_stats']:
                    stats['layer_stats'][layer] += 1
            
            # Check if sample is original or augmented
            if 'augmented' in sample['image_path'].name:
                stats['augmented'] += 1
            else:
                stats['original'] += 1
        
        return stats
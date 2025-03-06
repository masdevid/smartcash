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
    """Dataset for multi-layer banknote detection dengan konfigurasi terpusat."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        layers: Optional[List[str]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset directory
            img_size: Target image size
            mode: Dataset mode ('train', 'val', 'test')
            transform: Custom transformations
            layers: Layer yang akan diaktifkan (jika None, gunakan semua)
            logger: Logger instance
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.logger = logger or get_logger("multilayer_dataset", log_to_file=False)
        
        # Get layer configuration dari layer config manager
        self.layer_config_manager = get_layer_config()
        self.layers = layers or self.layer_config_manager.get_layer_names()
        
        # Setup paths
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        if not self.images_dir.exists():
            self.logger.warning(f"Images directory not found: {self.images_dir}")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
        if not self.labels_dir.exists():
            self.logger.warning(f"Labels directory not found: {self.labels_dir}")
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            
        self.image_files = self._find_image_files()
        
        if len(self.image_files) == 0:
            self.logger.warning(f"No images found in {self.images_dir}")
            
    def _find_image_files(self) -> List[Path]:
        """Find all image files with various extensions."""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(self.images_dir.glob(ext)))
        return sorted(image_files)
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def _normalize_bbox(self, bbox: List[float]) -> List[float]:
        """
        Normalize bounding box coordinates to ensure they're in [0, 1] range.
        
        Args:
            bbox: Bounding box coordinates [x_center, y_center, width, height]
            
        Returns:
            Normalized coordinates [x_center, y_center, width, height]
        """
        # Ensure x_center and y_center are in [0, 1]
        x_center = max(0.0, min(1.0, bbox[0]))
        y_center = max(0.0, min(1.0, bbox[1]))
        
        # Ensure width and height are valid
        width = max(0.01, min(1.0, bbox[2]))
        if x_center + width/2 > 1.0:
            width = 2 * (1.0 - x_center)
            
        height = max(0.01, min(1.0, bbox[3]))
        if y_center + height/2 > 1.0:
            height = 2 * (1.0 - y_center)
            
        return [x_center, y_center, width, height]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get dataset item dengan format multilayer yang lebih baik."""
        try:
            # Handle empty dataset
            if len(self.image_files) == 0:
                dummy_img = torch.zeros((3, self.img_size[1], self.img_size[0]))
                dummy_targets = {layer: torch.zeros((len(self.layer_config_manager.get_layer_config(layer)['classes']), 5)) 
                                for layer in self.layers}
                return dummy_img, dummy_targets
                
            img_path = self.image_files[idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            # Load and validate image
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning(f"Cannot read image: {img_path}")
                dummy_img = torch.zeros((3, self.img_size[1], self.img_size[0]))
                dummy_targets = {layer: torch.zeros((len(self.layer_config_manager.get_layer_config(layer)['classes']), 5)) 
                                for layer in self.layers}
                return dummy_img, dummy_targets
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Mapping class ID to layer
            class_to_layer = {}
            for layer in self.layers:
                layer_config = self.layer_config_manager.get_layer_config(layer)
                for cls_id in layer_config['class_ids']:
                    class_to_layer[cls_id] = layer
            
            # Load and validate labels
            bboxes = []
            class_labels = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                coords = list(map(float, parts[1:5]))
                                
                                # Validasi class ID dengan layer_config_manager
                                layer = self.layer_config_manager.get_layer_for_class_id(class_id)
                                if layer in self.layers:
                                    normalized_coords = self._normalize_bbox(coords)
                                    bboxes.append(normalized_coords)
                                    class_labels.append(class_id)
                except Exception as e:
                    self.logger.warning(f"Error reading label {label_path}: {str(e)}")
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            else:
                # Default transform (resize and normalize)
                transform = A.Compose([
                    A.Resize(height=self.img_size[1], width=self.img_size[0]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                
            # Validate transformed bboxes
            validated_bboxes = []
            validated_class_labels = []
            
            for bbox, cls_id in zip(transformed['bboxes'], transformed['class_labels']):
                try:
                    for coord in bbox:
                        if not (0 <= coord <= 1):
                            raise ValueError(f"Invalid bbox coordinate {bbox}")
                    validated_bboxes.append(bbox)
                    validated_class_labels.append(cls_id)
                except ValueError as e:
                    self.logger.warning(f"Skipping invalid bbox: {bbox}")
                    continue
                    
            # Update transformed data
            transformed['bboxes'] = validated_bboxes
            transformed['class_labels'] = validated_class_labels
            
            # Prepare final tensors
            img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
            
            # Ensure all class labels are integers
            class_labels = [int(cl) for cl in transformed['class_labels']]
            
            # Prepare layer-specific targets
            layer_targets = {}
            
            for layer in self.layers:
                layer_config = self.layer_config_manager.get_layer_config(layer)
                num_classes = len(layer_config['classes'])
                class_ids = layer_config['class_ids']
                
                # Initialize tensors for this layer
                layer_target = torch.zeros((num_classes, 5))  # [x, y, w, h, conf]
                
                # Find boxes for this layer
                for bbox, cls_id in zip(transformed['bboxes'], transformed['class_labels']):
                    if cls_id in class_ids:
                        # Compute index relative to this layer's class range
                        local_idx = class_ids.index(cls_id)
                        
                        # Fill in coordinates and confidence
                        x_center, y_center, width, height = bbox
                        layer_target[local_idx, 0] = x_center
                        layer_target[local_idx, 1] = y_center
                        layer_target[local_idx, 2] = width
                        layer_target[local_idx, 3] = height
                        layer_target[local_idx, 4] = 1.0  # Confidence
                
                layer_targets[layer] = layer_target
                
            return img_tensor, layer_targets
            
        except Exception as e:
            self.logger.error(f"Error loading item {idx}: {str(e)}")
            # Return zero tensors as fallback
            dummy_img = torch.zeros((3, self.img_size[1], self.img_size[0]))
            dummy_targets = {layer: torch.zeros((len(self.layer_config_manager.get_layer_config(layer)['classes']), 5)) 
                            for layer in self.layers}
            return dummy_img, dummy_targets

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
        shuffle: Optional[bool] = None
    ) -> DataLoader:
        """
        Get DataLoader for specific dataset split.
        
        Args:
            split: Dataset split ('train', 'valid', 'test')
            batch_size: Batch size
            num_workers: Number of workers
            shuffle: Whether to shuffle data (defults to True for train, False otherwise)
            
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
            
        # Create dataset
        dataset = MultilayerDataset(
            data_path=data_path,
            img_size=self.target_size,
            mode=split,
            transform=transform,
            layers=self.active_layers,
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
        """Collate function for multilayer dataset."""
        # Filter out None values
        batch = [b for b in batch if b is not None and isinstance(b, tuple) and len(b) == 2]
        
        if len(batch) == 0:
            # Return empty batch with layer structure
            dummy_targets = {layer: torch.zeros((0, len(self.layer_config.get_layer_config(layer)['classes']), 5))
                             for layer in self.active_layers}
            return torch.zeros((0, 3, self.target_size[0], self.target_size[1])), dummy_targets
            
        imgs, targets_list = zip(*batch)
        
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
        Dapatkan statistik dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Statistik dataset
        """
        # Determine path from config
        if split == 'train':
            data_path = self.config.get('data', {}).get('local', {}).get('train', 'data/train')
        elif split == 'valid' or split == 'val':
            data_path = self.config.get('data', {}).get('local', {}).get('valid', 'data/valid')
        elif split == 'test':
            data_path = self.config.get('data', {}).get('local', {}).get('test', 'data/test')
        else:
            raise ValueError(f"Invalid split: {split}")
            
        # Statistik dasar
        stats = {
            'image_count': 0,
            'label_count': 0,
            'layer_stats': {layer: 0 for layer in self.active_layers},
            'class_stats': {}
        }
        
        # Hitung jumlah gambar
        image_dir = Path(data_path) / 'images'
        if image_dir.exists():
            stats['image_count'] = len(list(image_dir.glob('*.jpg'))) + len(list(image_dir.glob('*.png')))
        
        # Hitung jumlah label
        label_dir = Path(data_path) / 'labels'
        if label_dir.exists():
            stats['label_count'] = len(list(label_dir.glob('*.txt')))
            
            # Analisis label
            class_to_layer = {}
            class_names = {}
            
            # Setup mapping dari class ID ke layer dan nama
            for layer in self.active_layers:
                layer_config = self.layer_config.get_layer_config(layer)
                for i, class_id in enumerate(layer_config['class_ids']):
                    class_to_layer[class_id] = layer
                    class_names[class_id] = layer_config['classes'][i]
            
            # Analisis file label
            for label_file in label_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                
                                # Update layer stats
                                if class_id in class_to_layer:
                                    layer = class_to_layer[class_id]
                                    stats['layer_stats'][layer] += 1
                                    
                                    # Update class stats
                                    class_name = class_names.get(class_id, f"unknown_{class_id}")
                                    if class_name not in stats['class_stats']:
                                        stats['class_stats'][class_name] = 0
                                    stats['class_stats'][class_name] += 1
                except Exception as e:
                    self.logger.warning(f"Error analyzing label {label_file}: {str(e)}")
        
        return stats
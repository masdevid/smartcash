# File: smartcash/handlers/data_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk pemrosesan dataset dengan validasi dan normalisasi koordinat bounding box

import os
import torch
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm.auto import tqdm
import logging

# Fungsi logging global untuk menghindari lambda
def default_log_info(msg):
    print(f"ℹ️ {msg}")

def default_log_warning(msg):
    print(f"⚠️ {msg}")

def default_log_error(msg):
    print(f"❌ {msg}")

def default_log_success(msg):
    print(f"✅ {msg}")

class MultilayerDataset(Dataset):
    """Dataset untuk deteksi multi-layer uang kertas Rupiah."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path ke direktori dataset
            img_size: Ukuran gambar yang diinginkan
            mode: Mode dataset ('train', 'val', 'test')
            transform: Custom transformations
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        
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
            default_log_warning(f"⚠️ No images found in {self.images_dir}")
            
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
            
        default_log_info(f"📚 Loaded {len(self.image_files)} images for {mode} dataset")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _normalize_bbox(self, bbox: List[float]) -> List[float]:
        """
        Normalisasi bounding box untuk memastikan semua koordinat dalam range [0, 1].
        
        Args:
            bbox: Koordinat bounding box [x_center, y_center, width, height]
            
        Returns:
            Koordinat yang sudah dinormalisasi [x_center, y_center, width, height]
        """
        # Pastikan x_center dan y_center berada dalam rentang [0, 1]
        x_center = max(0.0, min(1.0, bbox[0]))
        y_center = max(0.0, min(1.0, bbox[1]))
        
        # Pastikan width dan height berada dalam rentang yang logis
        # Width tidak lebih dari 1.0 dan x_center + width/2 tidak lebih dari 1.0
        width = max(0.01, min(1.0, bbox[2]))
        if x_center + width/2 > 1.0:
            width = 2 * (1.0 - x_center)
            
        # Height tidak lebih dari 1.0 dan y_center + height/2 tidak lebih dari 1.0
        height = max(0.01, min(1.0, bbox[3]))
        if y_center + height/2 > 1.0:
            height = 2 * (1.0 - y_center)
            
        return [x_center, y_center, width, height]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset dengan validasi koordinat bounding box."""
        try:
            img_path = self.image_files[idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                raise IOError(f"Tidak dapat membaca gambar: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load labels
            bboxes = []
            class_labels = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class_id, x, y, w, h
                            # PERBAIKAN: Pastikan class_id adalah integer
                            # dan koordinat bounding box adalah float
                            try:
                                class_id = int(float(parts[0]))  # Konversi ke int dengan aman
                                
                                # Baca koordinat dan lakukan normalisasi
                                coords = list(map(float, parts[1:5]))
                                normalized_coords = self._normalize_bbox(coords)
                                
                                bboxes.append(normalized_coords)
                                class_labels.append(class_id)
                            except (ValueError, IndexError) as e:
                                default_log_error(f"⚠️ Baris label tidak valid pada {label_path}: {line.strip()} - {str(e)}")
                                continue
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            else:
                transformed = self.default_transform(image=img, bboxes=bboxes, class_labels=class_labels)
                
            # PERBAIKAN: Validasi hasil transformasi
            validated_bboxes = []
            validated_class_labels = []
            
            for bbox, cls_id in zip(transformed['bboxes'], transformed['class_labels']):
                # Validasi koordinat
                try:
                    for coord in bbox:
                        if not (0 <= coord <= 1):
                            raise ValueError(f"Koordinat bbox {bbox} di luar rentang [0, 1]")
                    validated_bboxes.append(bbox)
                    validated_class_labels.append(cls_id)
                except ValueError as e:
                    default_log_warning(f"⚠️ Mengabaikan bbox tidak valid: {bbox} - {str(e)}")
                    continue
                
            # Update hasil transformasi
            transformed['bboxes'] = validated_bboxes
            transformed['class_labels'] = validated_class_labels
                
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
            # Convert to tensor format
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            
            # PERBAIKAN: Pastikan semua class_labels adalah integer
            class_labels = [int(cl) for cl in class_labels]
            
            # Prepare multi-layer targets
            total_classes = sum(len(layer['classes']) for layer in self.layer_config.values())
            targets = torch.zeros((total_classes, 5))  # [class_id, x, y, w, h]
            
            for i, (bbox, cls_id) in enumerate(zip(bboxes, class_labels)):
                # PERBAIKAN: Pastikan cls_id dalam range yang valid dan integer
                if 0 <= cls_id < total_classes:
                    x_center, y_center, width, height = bbox
                    targets[cls_id, 0] = x_center
                    targets[cls_id, 1] = y_center
                    targets[cls_id, 2] = width
                    targets[cls_id, 3] = height
                    targets[cls_id, 4] = 1.0  # Confidence
                else:
                    default_log_warning(f"⚠️ Class ID {cls_id} di luar jangkauan (0-{total_classes-1}) pada {img_path}")
                
            return img_tensor, targets
            
        except Exception as e:
            default_log_error(f"❌ Error loading item {idx}: {str(e)}")
            # Return a zero image and empty target as fallback
            img_tensor = torch.zeros((3, self.img_size[1], self.img_size[0]))
            total_classes = sum(len(layer['classes']) for layer in self.layer_config.values())
            targets = torch.zeros((total_classes, 5))
            return img_tensor, targets

class DataHandler:
    """Handler untuk data training dan evaluasi."""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        logger: Optional[Callable] = None
    ):
        """
        Inisialisasi data handler.
        
        Args:
            config: Konfigurasi dataset (opsional)
            data_dir: Direktori data (opsional)
            logger: Logger untuk output (opsional)
        """
        self.config = config or {}
        self.data_dir = data_dir or self.config.get('data_dir', 'data')
        
        # Setup logger functions
        self.log_info = logger.info if logger else default_log_info
        self.log_error = logger.error if logger else default_log_error
        self.log_success = logger.success if logger else default_log_success
        self.log_warning = logger.warning if logger else default_log_warning
        
        # Setup image size
        self.img_size = self.config.get('model', {}).get('img_size', [640, 640])
        
        # Setup dataset paths
        self._setup_dataset_paths()
    
    def _setup_dataset_paths(self) -> None:
        """Setup jalur dataset berdasarkan konfigurasi atau default."""
        if self.config.get('data_source') == 'local':
            # Gunakan jalur dari konfigurasi jika ada
            if 'data' in self.config and 'local' in self.config['data']:
                data_local = self.config['data']['local']
                self.train_path = data_local.get('train', os.path.join(self.data_dir, 'train'))
                self.val_path = data_local.get('valid', os.path.join(self.data_dir, 'valid'))
                if 'val' in data_local and 'valid' not in data_local:
                    self.val_path = data_local.get('val')
                self.test_path = data_local.get('test', os.path.join(self.data_dir, 'test'))
            else:
                # Jalur default
                self.train_path = os.path.join(self.data_dir, 'train')
                self.val_path = os.path.join(self.data_dir, 'valid')
                self.test_path = os.path.join(self.data_dir, 'test')
        else:
            # Default paths untuk kasus non-local
            self.train_path = self.config.get('train_data_path', os.path.join(self.data_dir, 'train'))
            self.val_path = self.config.get('val_data_path', os.path.join(self.data_dir, 'valid'))
            self.test_path = self.config.get('test_data_path', os.path.join(self.data_dir, 'test'))
        
        # Log jalur dataset
        self.log_info(f"📂 Dataset paths:")
        self.log_info(f"   Train: {self.train_path}")
        self.log_info(f"   Val: {self.val_path}")
        self.log_info(f"   Test: {self.test_path}")
    
    def setup_dataset_structure(self) -> None:
        """Siapkan struktur direktori dataset."""
        for path in [self.train_path, self.val_path, self.test_path]:
            # Buat direktori images dan labels
            for subdir in ['images', 'labels']:
                dir_path = os.path.join(path, subdir)
                os.makedirs(dir_path, exist_ok=True)
                self.log_info(f"✅ Direktori dibuat: {dir_path}")
    
    def get_train_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_path: Optional[str] = None
    ) -> DataLoader:
        """
        Dapatkan data loader untuk training.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            dataset_path: Custom dataset path (opsional)
            
        Returns:
            DataLoader untuk training
        """
        dataset_path = dataset_path or self.train_path
        self.log_info(f"🔄 Mempersiapkan training dataloader dari {dataset_path}")
        
        # Buat dataset
        dataset = MultilayerDataset(
            data_path=dataset_path,
            img_size=tuple(self.img_size),
            mode='train'
        )
        
        # Buat dan kembalikan dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def get_val_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_path: Optional[str] = None
    ) -> DataLoader:
        """
        Dapatkan data loader untuk validasi.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            dataset_path: Custom dataset path (opsional)
            
        Returns:
            DataLoader untuk validasi
        """
        dataset_path = dataset_path or self.val_path
        self.log_info(f"🔄 Mempersiapkan validation dataloader dari {dataset_path}")
        
        # Buat dataset
        dataset = MultilayerDataset(
            data_path=dataset_path,
            img_size=tuple(self.img_size),
            mode='val'
        )
        
        # Buat dan kembalikan dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def get_test_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_path: Optional[str] = None
    ) -> DataLoader:
        """
        Dapatkan data loader untuk testing.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            dataset_path: Custom dataset path (opsional)
            
        Returns:
            DataLoader untuk testing
        """
        dataset_path = dataset_path or self.test_path
        self.log_info(f"🔄 Mempersiapkan test dataloader dari {dataset_path}")
        
        # Buat dataset
        dataset = MultilayerDataset(
            data_path=dataset_path,
            img_size=tuple(self.img_size),
            mode='test'
        )
        
        # Buat dan kembalikan dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def get_class_names(self) -> List[str]:
        """
        Dapatkan nama kelas dari dataset.
        
        Returns:
            List nama kelas
        """
        # Coba ambil dari konfigurasi
        class_names = self.config.get('dataset', {}).get('classes')
        if class_names:
            return class_names
            
        # Fallback ke default class names
        return ['001', '002', '005', '010', '020', '050', '100']
    
    def _collate_fn(self, batch):
        """
        Custom collate function untuk batching items.
        
        Args:
            batch: Batch dari dataset
            
        Returns:
            Tuple (img_tensor, targets_tensor)
        """
        imgs, targets = zip(*batch)
        
        # Batch images - sudah dalam format tensor dari dataset.__getitem__
        imgs = torch.stack(imgs)
        
        # Batch targets - perlu handling khusus untuk format multilayer
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets)
        
        return imgs, targets
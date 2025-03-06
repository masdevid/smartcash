# File: smartcash/handlers/multilayer_dataset_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk menangani dataset multilayer dengan dukungan validasi dan pengecekan konsistensi layer

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm.auto import tqdm
import logging

from smartcash.utils.logger import SmartCashLogger

class MultilayerDataset(Dataset):
    """Dataset untuk deteksi multilayer dengan penanganan layer yang tidak lengkap."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        layers: List[str] = ['banknote', 'nominal', 'security'],
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
        self.layers = layers
        self.require_all_layers = require_all_layers
        self.logger = logger or SmartCashLogger("multilayer_dataset")
        
        # Konfigurasi layer
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
                    for layer, config in self.layer_config.items():
                        class_ids = set(config['class_ids'])
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
        layer_counts = {layer: 0 for layer in self.layer_config.keys()}
        for sample in valid_samples:
            for layer in sample['available_layers']:
                layer_counts[layer] += 1
                
        for layer, count in layer_counts.items():
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
        
        # Mapping class ID ke layer
        class_to_layer = {}
        for layer, config in self.layer_config.items():
            for cls_id in config['class_ids']:
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
                            
                            # Cek apakah class ID termasuk dalam layer yang diaktifkan
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
                
                for i, (cls_id, bbox) in enumerate(zip(transformed['class_labels'], transformed['bboxes'])):
                    for layer, config in self.layer_config.items():
                        if cls_id in config['class_ids'] and layer in self.layers:
                            augmented_layer_labels[layer].append((cls_id, bbox))
                            break
                
                layer_labels = augmented_layer_labels
                
            except Exception as e:
                self.logger.warning(f"⚠️ Augmentasi gagal untuk {sample['image_path']}: {str(e)}")
                # Lanjutkan dengan label asli
                pass
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
                layer_config = self.layer_config[layer]
                num_classes = len(layer_config['classes'])
                
                # Inisialisasi target untuk layer ini
                layer_target = torch.zeros((num_classes, 5))  # [class_id, x, y, w, h]
                
                # Isi dengan data label
                for cls_id, bbox in layer_labels[layer]:
                    # Konversi kembali class ID global ke local dalam layer
                    local_cls_id = cls_id - min(layer_config['class_ids'])
                    if 0 <= local_cls_id < num_classes:
                        x_center, y_center, width, height = bbox
                        layer_target[local_cls_id, 0] = x_center
                        layer_target[local_cls_id, 1] = y_center
                        layer_target[local_cls_id, 2] = width
                        layer_target[local_cls_id, 3] = height
                        layer_target[local_cls_id, 4] = 1.0  # Confidence
                
                targets[layer] = layer_target
            else:
                # Layer tidak tersedia, buat tensor kosong
                num_classes = len(self.layer_config[layer]['classes'])
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

class MultilayerDataManager:
    """Manager untuk dataset multilayer dengan transformasi yang konsisten."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi MultilayerDataManager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom
        """
        self.config = config
        self.logger = logger or SmartCashLogger("multilayer_data_manager")
        
        # Ambil parameter dari config
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        self.batch_size = config.get('model', {}).get('batch_size', 16)
        self.num_workers = config.get('model', {}).get('workers', 4)
        
        # Ambil layer yang diaktifkan
        self.layers = config.get('layers', ['banknote'])
        
        # Setup transformasi
        self._setup_transformations()
        
    def _setup_transformations(self):
        """Setup transformasi untuk training dan validasi."""
        train_config = self.config.get('training', {})
        
        # Transformasi untuk training
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=self.img_size[1],
                width=self.img_size[0],
                scale=(0.8, 1.0),
                p=1.0
            ),
            A.HorizontalFlip(p=train_config.get('fliplr', 0.5)),
            A.VerticalFlip(p=train_config.get('flipud', 0.0)),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=train_config.get('hsv_h', 0.015),
                sat_shift_limit=train_config.get('hsv_s', 0.7),
                val_shift_limit=train_config.get('hsv_v', 0.4),
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=train_config.get('translate', 0.1),
                scale_limit=train_config.get('scale', 0.5),
                rotate_limit=train_config.get('degrees', 0.0),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Transformasi untuk validasi
        self.val_transform = A.Compose([
            A.Resize(height=self.img_size[1], width=self.img_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
    def get_dataset(
        self,
        mode: str = 'train',
        require_all_layers: bool = False
    ) -> MultilayerDataset:
        """
        Mendapatkan dataset berdasarkan mode.
        
        Args:
            mode: Mode dataset ('train', 'val', 'test')
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            
        Returns:
            MultilayerDataset instance
        """
        # Tentukan path data
        if mode == 'train':
            data_path = self.config.get('data', {}).get('local', {}).get('train', 'data/train')
            transform = self.train_transform
        elif mode == 'val' or mode == 'valid':
            data_path = self.config.get('data', {}).get('local', {}).get('valid', 'data/valid')
            transform = self.val_transform
        elif mode == 'test':
            data_path = self.config.get('data', {}).get('local', {}).get('test', 'data/test')
            transform = self.val_transform
        else:
            raise ValueError(f"Mode tidak valid: {mode}")
        
        return MultilayerDataset(
            data_path=data_path,
            img_size=self.img_size,
            mode=mode,
            transform=transform,
            layers=self.layers,
            require_all_layers=require_all_layers,
            logger=self.logger
        )
        
    def get_dataloader(
        self,
        mode: str = 'train',
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        require_all_layers: bool = False,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Mendapatkan dataloader untuk dataset multilayer.
        
        Args:
            mode: Mode dataset ('train', 'val', 'test')
            batch_size: Ukuran batch (jika None, menggunakan nilai dari config)
            num_workers: Jumlah worker (jika None, menggunakan nilai dari config)
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            pin_memory: Apakah menggunakan pin_memory
            
        Returns:
            DataLoader instance
        """
        # Gunakan nilai default dari config jika tidak diberikan
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        
        # Buat dataset
        dataset = self.get_dataset(mode, require_all_layers)
        
        # Buat dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=self._multilayer_collate_fn,
            drop_last=mode == 'train'
        )
    
    def _multilayer_collate_fn(self, batch):
        """Custom collate function untuk data multilayer."""
        if len(batch) == 0:
            return None
        
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        # Ekstrak images dan targets
        images = [item['image'] for item in batch]
        targets = [item['targets'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        # Stack images
        batched_images = torch.stack(images)
        
        # Batch targets per layer
        batched_targets = {}
        for layer in self.layers:
            layer_targets = [t[layer] for t in targets if layer in t]
            if layer_targets:
                batched_targets[layer] = torch.stack(layer_targets)
            else:
                # Jika tidak ada target untuk layer ini, buat tensor kosong
                num_classes = len(self.config.get('layer_config', {}).get(layer, {}).get(
                    'classes', []
                ))
                if num_classes == 0:
                    # Fallback jika config tidak memiliki informasi kelas
                    if layer == 'banknote':
                        num_classes = 7
                    elif layer == 'nominal':
                        num_classes = 7
                    elif layer == 'security':
                        num_classes = 3
                    else:
                        num_classes = 1
                
                batched_targets[layer] = torch.zeros(
                    (len(batch), num_classes, 5),
                    device=batched_images.device
                )
        
        return {
            'images': batched_images,
            'targets': batched_targets,
            'metadata': metadata
        }
    
    def get_dataset_stats(self) -> Dict:
        """
        Mendapatkan statistik dataset.
        
        Returns:
            Dict statistik dataset
        """
        stats = {}
        
        for mode in ['train', 'valid', 'test']:
            try:
                dataset = self.get_dataset(mode)
                
                # Hitung distribusi layer
                layer_counts = {layer: 0 for layer in self.layers}
                for sample in dataset.valid_samples:
                    for layer in sample['available_layers']:
                        if layer in self.layers:
                            layer_counts[layer] += 1
                
                stats[mode] = {
                    'total_samples': len(dataset),
                    'valid_samples': len(dataset.valid_samples),
                    'layer_distribution': layer_counts
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal mendapatkan statistik untuk {mode}: {str(e)}")
                stats[mode] = {'error': str(e)}
        
        return stats
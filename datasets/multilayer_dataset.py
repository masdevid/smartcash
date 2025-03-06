# File: smartcash/datasets/multilayer_dataset.py
# Author: Alfrida Sabar
# Deskripsi: Dataset multilayer yang terintegrasi dengan layer config manager untuk deteksi uang kertas

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
import albumentations as A
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
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
"""
File: smartcash/dataset/components/datasets/multilayer_dataset.py
Deskripsi: Implementasi dataset multilayer untuk menangani deteksi objek dengan multiple layer
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from torch.utils.data import Dataset
from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config


class MultilayerDataset(Dataset):
    """Dataset untuk deteksi objek dengan multiple layer."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        require_all_layers: bool = False,
        layers: Optional[List[str]] = None,
        logger = None,
        config: Optional[Dict] = None
    ):
        """
        Inisialisasi MultilayerDataset.
        
        Args:
            data_path: Path ke direktori data
            img_size: Ukuran gambar target
            mode: Mode dataset ('train', 'valid', 'test')
            transform: Transformasi yang akan diterapkan
            require_all_layers: Apakah memerlukan semua layer dalam setiap gambar
            layers: Daftar layer yang akan digunakan (opsional)
            logger: Logger kustom (opsional)
            config: Konfigurasi aplikasi (opsional)
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.require_all_layers = require_all_layers
        self.config = config or {}
        self.logger = logger or get_logger("multilayer_dataset")
        
        # Setup layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = layers or self.layer_config_manager.get_layer_names()
        
        # Setup directories
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        # Periksa direktori
        if not self.images_dir.exists() or not self.labels_dir.exists():
            self.logger.warning(
                f"⚠️ Direktori data tidak lengkap:\n"
                f"   • Image dir: {self.images_dir} {'✅' if self.images_dir.exists() else '❌'}\n"
                f"   • Label dir: {self.labels_dir} {'✅' if self.labels_dir.exists() else '❌'}"
            )
        
        # Membangun mapping class ID ke layer
        self.class_to_layer = {}
        self.class_to_name = {}
        
        for layer in self.active_layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            for i, cls_id in enumerate(layer_config['class_ids']):
                self.class_to_layer[cls_id] = layer
                if i < len(layer_config['classes']):
                    self.class_to_name[cls_id] = layer_config['classes'][i]
        
        # Cari semua file gambar dan label yang valid
        self.valid_samples = self._load_valid_samples()
        self.logger.info(f"✅ Dataset '{mode}' siap dengan {len(self.valid_samples)} sampel valid")
    
    def __len__(self) -> int:
        """Mendapatkan jumlah sampel dalam dataset."""
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Mengambil item dataset berdasarkan indeks.
        
        Args:
            idx: Indeks item yang akan diambil
            
        Returns:
            Dictionary berisi data sampel
        """
        if idx >= len(self.valid_samples):
            raise IndexError(f"Indeks {idx} melebihi ukuran dataset ({len(self.valid_samples)})")
        
        sample = self.valid_samples[idx]
        img_path = sample['image_path']
        label_path = sample['label_path']
        
        # Load dan proses gambar
        try:
            img = self._load_image(img_path)
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {img_path}: {str(e)}")
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # Load dan proses label
        layer_labels = self._parse_label_by_layer(label_path)
        
        # Flatten label untuk proses augmentasi
        all_bboxes, all_classes = [], []
        for layer_name, boxes in layer_labels.items():
            for cls_id, bbox in boxes:
                all_bboxes.append(bbox)
                all_classes.append(cls_id)
        
        # Terapkan transformasi jika ada
        if self.transform and len(all_bboxes) > 0:
            try:
                transformed = self.transform(
                    image=img,
                    bboxes=all_bboxes,
                    class_labels=all_classes
                )
                
                img = transformed['image']
                
                # Reorganisasi hasil transformasi kembali ke struktur per layer
                if 'bboxes' in transformed and 'class_labels' in transformed:
                    transformed_layer_labels = {layer: [] for layer in self.active_layers}
                    
                    for cls_id, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        if cls_id in self.class_to_layer:
                            layer = self.class_to_layer[cls_id]
                            transformed_layer_labels[layer].append((cls_id, bbox))
                    
                    layer_labels = transformed_layer_labels
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat transformasi: {str(e)}")
        elif self.transform:
            # Jika tidak ada bbox tetapi ada transform
            try:
                transformed = self.transform(image=img)
                img = transformed['image']
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat transformasi tanpa bbox: {str(e)}")
        
        # Konversi gambar ke tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        
        # Buat tensor target per layer
        targets = {}
        for layer in self.active_layers:
            layer_boxes = layer_labels.get(layer, [])
            targets[layer] = self._create_layer_tensor(layer, layer_boxes)
        
        return {
            'image': img_tensor,
            'targets': targets,
            'metadata': {
                'image_path': str(img_path),
                'label_path': str(label_path),
                'available_layers': sample['available_layers']
            }
        }
    
    def _load_valid_samples(self) -> List[Dict]:
        """
        Temukan semua sampel valid dalam dataset.
        
        Returns:
            List sampel valid dengan path dan metadata
        """
        valid_samples = []
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        # Cari semua file gambar
        for ext in img_extensions:
            for img_path in self.images_dir.glob(f'*{ext}'):
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    continue
                
                # Cek available layers
                available_layers = self._get_available_layers(label_path)
                
                # Jika memerlukan semua layer, pastikan semua active layer ada
                if self.require_all_layers and not all(layer in available_layers for layer in self.active_layers):
                    continue
                
                # Jika tidak memerlukan semua layer, pastikan setidaknya ada satu layer
                if not self.require_all_layers and not any(layer in available_layers for layer in self.active_layers):
                    continue
                
                valid_samples.append({
                    'image_path': img_path,
                    'label_path': label_path,
                    'available_layers': available_layers
                })
        
        return valid_samples
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load gambar dari file.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Array NumPy berisi gambar
        """
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize jika ukuran tidak sesuai dan tidak ada transform
        if not self.transform and (img.shape[0] != self.img_size[1] or img.shape[1] != self.img_size[0]):
            img = cv2.resize(img, self.img_size)
            
        return img
    
    def _parse_label_by_layer(self, label_path: Path) -> Dict[str, List[Tuple[int, List[float]]]]:
        """
        Parse file label dan kelompokkan berdasarkan layer.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            Dictionary berisi bounding box per layer
        """
        layer_labels = {layer: [] for layer in self.active_layers}
        
        if not label_path.exists():
            return layer_labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            x, y, w, h = map(float, parts[1:5])
                            
                            # Validasi koordinat
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                                if cls_id in self.class_to_layer:
                                    layer = self.class_to_layer[cls_id]
                                    layer_labels[layer].append((cls_id, [x, y, w, h]))
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return layer_labels
    
    def _get_available_layers(self, label_path: Path) -> List[str]:
        """
        Dapatkan daftar layer yang tersedia dalam file label.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List layer yang tersedia
        """
        available_layers = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            if cls_id in self.class_to_layer:
                                layer = self.class_to_layer[cls_id]
                                if layer not in available_layers:
                                    available_layers.append(layer)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
            
        return available_layers
    
    def _create_layer_tensor(self, layer: str, boxes: List[Tuple[int, List[float]]]) -> torch.Tensor:
        """
        Buat tensor untuk layer tertentu dari data label.
        
        Args:
            layer: Nama layer
            boxes: Daftar bounding box dan class ID untuk layer tersebut
            
        Returns:
            Tensor untuk layer
        """
        layer_config = self.layer_config_manager.get_layer_config(layer)
        num_classes = len(layer_config['classes'])
        class_ids = layer_config['class_ids']
        
        # Inisialisasi tensor kosong
        layer_tensor = torch.zeros((num_classes, 5))  # [x, y, w, h, conf]
        
        # Isi dengan data label
        for cls_id, bbox in boxes:
            if cls_id in class_ids:
                local_idx = class_ids.index(cls_id)
                if 0 <= local_idx < num_classes:
                    x_center, y_center, width, height = bbox
                    layer_tensor[local_idx, 0] = x_center
                    layer_tensor[local_idx, 1] = y_center
                    layer_tensor[local_idx, 2] = width
                    layer_tensor[local_idx, 3] = height
                    layer_tensor[local_idx, 4] = 1.0  # Confidence
                    
        return layer_tensor
    
    def get_layer_statistics(self) -> Dict[str, int]:
        """
        Dapatkan statistik jumlah objek per layer.
        
        Returns:
            Dictionary berisi jumlah objek per layer
        """
        stats = {layer: 0 for layer in self.active_layers}
        
        for sample in self.valid_samples:
            for layer in sample['available_layers']:
                if layer in stats:
                    stats[layer] += 1
                    
        return stats
    
    def get_class_statistics(self) -> Dict[str, int]:
        """
        Dapatkan statistik jumlah objek per kelas.
        
        Returns:
            Dictionary berisi jumlah objek per kelas
        """
        stats = {}
        
        for sample in self.valid_samples:
            label_path = sample['label_path']
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                if cls_id in self.class_to_name:
                                    class_name = self.class_to_name[cls_id]
                                    stats[class_name] = stats.get(class_name, 0) + 1
                            except (ValueError, IndexError):
                                continue
            except Exception:
                pass
                
        return stats
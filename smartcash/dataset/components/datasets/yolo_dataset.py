"""
File: smartcash/dataset/components/datasets/yolo_dataset.py
Deskripsi: Implementasi dataset YOLO untuk deteksi objek standar (non-multilayer)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.dataset.components.datasets.base_dataset import BaseDataset


class YOLODataset(BaseDataset):
    """Dataset untuk deteksi objek dengan format YOLO standar."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        class_names: Optional[List[str]] = None,
        logger = None,
        config: Optional[Dict] = None
    ):
        """
        Inisialisasi YOLODataset.
        
        Args:
            data_path: Path ke direktori data
            img_size: Ukuran gambar target
            mode: Mode dataset ('train', 'valid', 'test')
            transform: Transformasi yang akan diterapkan
            class_names: Daftar nama kelas (opsional)
            logger: Logger kustom (opsional)
            config: Konfigurasi aplikasi (opsional)
        """
        self.class_names = class_names or []
        super().__init__(data_path, img_size, mode, transform, logger, config)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Mengambil item dataset berdasarkan indeks.
        
        Args:
            idx: Indeks item yang akan diambil
            
        Returns:
            Dictionary berisi data sampel
        """
        if idx >= len(self.samples):
            raise IndexError(f"Indeks {idx} melebihi ukuran dataset ({len(self.samples)})")
        
        sample = self.samples[idx]
        img_path = sample['image_path']
        label_path = sample['label_path']
        
        # Load dan proses gambar
        try:
            img = self._load_image(img_path)
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {img_path}: {str(e)}")
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # Load dan proses label
        labels = self._parse_yolo_label(label_path)
        
        # Ekstrak bboxes dan class_ids untuk transformasi
        bboxes = [label['bbox'] for label in labels]
        class_ids = [label['class_id'] for label in labels]
        
        # Terapkan transformasi jika ada
        if self.transform and len(bboxes) > 0:
            try:
                transformed = self.transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_ids
                )
                
                img = transformed['image']
                
                if 'bboxes' in transformed and 'class_labels' in transformed:
                    bboxes = transformed['bboxes']
                    class_ids = transformed['class_labels']
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
        
        # Persiapkan targets dalam format YOLO
        targets = []
        for i, (bbox, cls_id) in enumerate(zip(bboxes, class_ids)):
            # Format YOLO: [class_id, x_center, y_center, width, height]
            target = torch.zeros(5)
            target[0] = cls_id
            target[1:5] = torch.tensor(bbox)
            targets.append(target)
        
        if targets:
            targets = torch.stack(targets)
        else:
            targets = torch.zeros((0, 5))
        
        return {
            'image': img_tensor,
            'targets': targets,
            'metadata': {
                'image_path': str(img_path),
                'label_path': str(label_path),
                'image_id': sample.get('image_id', img_path.stem)
            }
        }
    
    def _load_samples(self) -> List[Dict]:
        """
        Temukan semua sampel valid dalam dataset.
        
        Returns:
            List sampel valid dengan path dan metadata
        """
        valid_samples = []
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        # Cari semua file gambar
        for img_path in self.images_dir.glob('*.*'):
            if any(img_path.suffix.lower() == ext for ext in img_extensions):
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    continue
                
                # Pastikan label tidak kosong
                if label_path.stat().st_size == 0:
                    continue
                
                valid_samples.append({
                    'image_path': img_path,
                    'label_path': label_path,
                    'image_id': img_path.stem
                })
        
        return valid_samples
    
    def _parse_yolo_label(self, label_path: Path) -> List[Dict]:
        """
        Parse file label YOLO.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List berisi data bounding box dan kelas
        """
        if not label_path.exists():
            return []
        
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            bbox = list(map(float, parts[1:5]))  # [x_center, y_center, width, height]
                            
                            # Validasi koordinat
                            if not all(0 <= coord <= 1 for coord in bbox[0:2]) or not all(0 < coord <= 1 for coord in bbox[2:4]):
                                continue
                            
                            label = {
                                'class_id': cls_id,
                                'bbox': bbox
                            }
                            
                            # Tambahkan nama kelas jika ada
                            if self.class_names and 0 <= cls_id < len(self.class_names):
                                label['class_name'] = self.class_names[cls_id]
                            
                            labels.append(label)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        return labels
    
    def get_class_statistics(self) -> Dict[int, int]:
        """
        Dapatkan statistik jumlah objek per kelas.
        
        Returns:
            Dictionary berisi jumlah objek per kelas
        """
        class_counts = {}
        
        for sample in self.samples:
            label_path = sample['label_path']
            labels = self._parse_yolo_label(label_path)
            
            for label in labels:
                cls_id = label['class_id']
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                
        return class_counts
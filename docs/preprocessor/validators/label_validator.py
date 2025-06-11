"""
Validator untuk file label YOLO dalam dataset.

Modul ini menyediakan fungsionalitas untuk memvalidasi file label YOLO
sebelum diproses lebih lanjut dalam pipeline preprocessing.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set

from smartcash.common.logger import get_logger


class LabelValidator:
    """Validator untuk memeriksa format dan isi file label YOLO."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inisialisasi LabelValidator dengan konfigurasi.
        
        Args:
            config: Konfigurasi validator (opsional)
        """
        self.config = config or {}
        self.logger = get_logger()
        
        # Konfigurasi default
        self.min_objects = self.config.get('min_objects_per_image', 1)
        self.max_objects = self.config.get('max_objects_per_image', 100)
        self.allowed_classes = self.config.get('allowed_classes', None)  # None berarti semua kelas diizinkan
        self.require_normalized = self.config.get('require_normalized', True)
        self.min_bbox_size = self.config.get('min_bbox_size', 2)  # pixel
    
    def validate(self, label_path: Path, image_size: Optional[Tuple[int, int]] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validasi file label YOLO.
        
        Args:
            label_path: Path ke file label YOLO yang akan divalidasi
            image_size: Tuple (width, height) dari gambar terkait (opsional)
            
        Returns:
            Tuple berisi:
                - bool: True jika valid, False jika tidak
                - List[str]: Daftar pesan error (jika ada)
                - Dict[str, Any]: Statistik validasi
        """
        errors = []
        stats = {
            'num_objects': 0,
            'class_distribution': {},
            'bbox_sizes': [],
            'is_empty': False,
            'has_invalid_lines': 0
        }
        
        try:
            # Periksa apakah file label ada
            if not label_path.exists() or label_path.stat().st_size == 0:
                errors.append("File label kosong atau tidak ditemukan")
                stats['is_empty'] = True
                return False, errors, stats
            
            # Baca isi file label
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                errors.append("File label kosong")
                stats['is_empty'] = True
                return False, errors, stats
            
            # Inisialisasi distribusi kelas
            class_dist = {}
            if self.allowed_classes is not None:
                for cls_id in self.allowed_classes:
                    class_dist[cls_id] = 0
            
            # Validasi setiap baris
            valid_objects = 0
            invalid_lines = 0
            bbox_sizes = []
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Validasi format baris
                parts = line.split()
                if len(parts) < 5:  # Minimal: class_id x y w h
                    errors.append(f"Format tidak valid di baris {i}: '{line}'")
                    invalid_lines += 1
                    continue
                
                # Validasi class_id
                try:
                    class_id = int(parts[0])
                    if self.allowed_classes is not None and class_id not in self.allowed_classes:
                        errors.append(f"Kelas ID {class_id} tidak diizinkan di baris {i}")
                        continue
                except ValueError:
                    errors.append(f"Class ID harus berupa angka di baris {i}: '{parts[0]}'")
                    invalid_lines += 1
                    continue
                
                # Validasi koordinat bbox (x, y, w, h)
                try:
                    coords = [float(x) for x in parts[1:5]]
                    x, y, w, h = coords
                    
                    # Validasi nilai normalisasi (jika diperlukan)
                    if self.require_normalized:
                        if any(not (0.0 <= val <= 1.0) for val in [x, y, w, h]):
                            errors.append(f"Koordinat harus dinormalisasi [0-1] di baris {i}")
                            continue
                    
                    # Hitung ukuran bbox dalam piksel (jika image_size disediakan)
                    if image_size is not None:
                        img_w, img_h = image_size
                        bbox_w = w * img_w if self.require_normalized else w
                        bbox_h = h * img_h if self.require_normalized else h
                        bbox_area = bbox_w * bbox_h
                        bbox_sizes.append({
                            'width': bbox_w,
                            'height': bbox_h,
                            'area': bbox_area,
                            'class_id': class_id
                        })
                        
                        # Validasi ukuran bbox minimum
                        if bbox_w < self.min_bbox_size or bbox_h < self.min_bbox_size:
                            errors.append(
                                f"Bounding box terlalu kecil ({int(bbox_w)}x{int(bbox_h)}px) "
                                f"untuk kelas {class_id} di baris {i} (min: {self.min_bbox_size}px)"
                            )
                    
                    # Update distribusi kelas
                    class_dist[class_id] = class_dist.get(class_id, 0) + 1
                    valid_objects += 1
                    
                except (ValueError, IndexError) as e:
                    errors.append(f"Format koordinat tidak valid di baris {i}: {str(e)}")
                    invalid_lines += 1
            
            # Update statistik
            stats.update({
                'num_objects': valid_objects,
                'class_distribution': class_dist,
                'bbox_sizes': bbox_sizes,
                'has_invalid_lines': invalid_lines
            })
            
            # Validasi jumlah objek
            if valid_objects < self.min_objects:
                errors.append(
                    f"Jumlah objek ({valid_objects}) kurang dari minimum yang dibutuhkan ({self.min_objects})"
                )
            
            if valid_objects > self.max_objects:
                errors.append(
                    f"Jumlah objek ({valid_objects}) melebihi batas maksimum ({self.max_objects})"
                )
            
            # Validasi distribusi kelas (jika ada kelas yang diwajibkan)
            if self.allowed_classes is not None:
                missing_classes = [str(cls) for cls in self.allowed_classes if class_dist.get(cls, 0) == 0]
                if missing_classes:
                    errors.append(f"Kelas yang diperlukan tidak ditemukan: {', '.join(missing_classes)}")
            
            return len(errors) == 0, errors, stats
            
        except Exception as e:
            error_msg = f"Error saat memvalidasi {label_path.name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return False, errors, stats
    
    def batch_validate(self, label_paths: List[Path], image_sizes: Optional[List[Tuple[int, int]]] = None) -> Dict[Path, Dict[str, Any]]:
        """Validasi sekumpulan file label YOLO.
        
        Args:
            label_paths: Daftar path ke file label YOLO yang akan divalidasi
            image_sizes: Daftar tuple (width, height) untuk setiap gambar (opsional)
            
        Returns:
            Dictionary dengan path label sebagai key dan dictionary berisi:
            - valid: bool
            - errors: List[str]
            - stats: Dict[str, Any]
        """
        results = {}
        
        if image_sizes is not None and len(image_sizes) != len(label_paths):
            self.logger.warning("Jumlah image_sizes tidak sesuai dengan label_paths, mengabaikan image_sizes")
            image_sizes = None
        
        for i, label_path in enumerate(label_paths):
            img_size = image_sizes[i] if image_sizes else None
            is_valid, errors, stats = self.validate(label_path, img_size)
            results[label_path] = {
                'valid': is_valid,
                'errors': errors,
                'stats': stats
            }
        
        return results


def create_label_validator(config: Dict[str, Any] = None) -> LabelValidator:
    """Factory function untuk membuat instance LabelValidator."""
    return LabelValidator(config)

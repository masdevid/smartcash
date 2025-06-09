"""
SmartCash Validation Engine

Modul untuk validasi dataset YOLOv5 sebelum preprocessing.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.file_scanner import FileScanner
from smartcash.dataset.augmentor.utils.path_resolver import PathResolver


class ValidationEngine:
    """Engine untuk validasi dataset YOLOv5."""
    
    def __init__(self, config: Dict[str, Any]):
        """Inisialisasi validation engine.
        
        Args:
            config: Konfigurasi validasi
        """
        self.config = config
        self.logger = get_logger()
        
        # Inisialisasi komponen
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(config)
        
        # Konfigurasi default
        self.min_img_size = config.get('min_image_size', 32)
        self.max_img_size = config.get('max_image_size', 4096)
        self.allowed_img_ext = ['.jpg', '.jpeg', '.png']
        self.required_classes = set(config.get('classes', []))
    
    def validate_split(self, split: str) -> Dict[str, Any]:
        """Validasi dataset untuk split tertentu.
        
        Args:
            split: Nama split yang akan divalidasi (train/val/test)
            
        Returns:
            Dictionary berisi hasil validasi dan statistik
        """
        try:
            # Dapatkan path direktori
            img_dir = self.path_resolver.get_source_image_dir(split)
            label_dir = self.path_resolver.get_source_label_dir(split)
            
            # Periksa keberadaan direktori
            if not img_dir.exists() or not label_dir.exists():
                return {
                    'is_valid': False,
                    'message': f"Direktori sumber tidak ditemukan: {img_dir} atau {label_dir}",
                    'stats': {}
                }
            
            # Dapatkan daftar file
            img_files = self.file_scanner.scan_directory(img_dir, self.allowed_img_ext)
            
            if not img_files:
                return {
                    'is_valid': False,
                    'message': f"Tidak ada file gambar yang ditemukan di {img_dir}",
                    'stats': {'total_images': 0}
                }
            
            # Inisialisasi statistik
            stats = {
                'total_images': len(img_files),
                'valid_images': 0,
                'invalid_images': 0,
                'missing_labels': 0,
                'invalid_labels': 0,
                'class_distribution': {},
                'image_sizes': [],
                'validation_errors': []
            }
            
            # Validasi setiap gambar
            for img_file in img_files:
                img_valid, img_errors = self._validate_image(img_file)
                
                # Dapatkan file label yang sesuai
                label_file = self._get_corresponding_label_file(img_file, label_dir)
                label_valid, label_errors, label_stats = self._validate_label(label_file, img_file)
                
                # Update statistik
                if img_valid and label_valid:
                    stats['valid_images'] += 1
                    
                    # Update distribusi kelas
                    for cls_id in label_stats.get('class_ids', []):
                        stats['class_distribution'][cls_id] = stats['class_distribution'].get(cls_id, 0) + 1
                    
                    # Update ukuran gambar
                    stats['image_sizes'].append(label_stats.get('image_size', (0, 0)))
                else:
                    stats['invalid_images'] += 1
                    stats['validation_errors'].extend(img_errors)
                    stats['validation_errors'].extend(label_errors)
                
                if not label_file:
                    stats['missing_labels'] += 1
                elif not label_valid:
                    stats['invalid_labels'] += 1
            
            # Hitung statistik tambahan
            stats['validation_passed'] = stats['invalid_images'] == 0
            
            # Hitung ukuran gambar rata-rata
            if stats['image_sizes']:
                avg_width = sum(w for w, _ in stats['image_sizes']) / len(stats['image_sizes'])
                avg_height = sum(h for _, h in stats['image_sizes']) / len(stats['image_sizes'])
                stats['avg_image_size'] = (int(avg_width), int(avg_height))
            
            return {
                'is_valid': stats['validation_passed'],
                'message': 'Validasi berhasil' if stats['validation_passed'] else 'Ditemukan masalah pada dataset',
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error dalam validasi {split}: {str(e)}", exc_info=True)
            return {
                'is_valid': False,
                'message': f"Error dalam validasi: {str(e)}",
                'stats': {}
            }
    
    def _validate_image(self, img_path: Path) -> Tuple[bool, List[str]]:
        """Validasi file gambar."""
        errors = []
        
        try:
            # Periksa ukuran file
            file_size = os.path.getsize(img_path) / (1024 * 1024)  # Dalam MB
            if file_size > 10:  # Lebih dari 10MB
                errors.append(f"Ukuran file {img_path.name} terlalu besar: {file_size:.2f}MB")
            
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Gagal membaca gambar: {img_path.name}")
                return False, errors
            
            # Periksa dimensi gambar
            h, w = img.shape[:2]
            if h < self.min_img_size or w < self.min_img_size:
                errors.append(
                    f"Gambar {img_path.name} terlalu kecil: {w}x{h} "
                    f"(minimum: {self.min_img_size}x{self.min_img_size})"
                )
            
            if h > self.max_img_size or w > self.max_img_size:
                errors.append(
                    f"Gambar {img_path.name} terlalu besar: {w}x{h} "
                    f"(maksimum: {self.max_img_size}x{self.max_img_size})"
                )
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Error saat memvalidasi {img_path.name}: {str(e)}")
            return False, errors
    
    def _validate_label(self, label_path: Optional[Path], img_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validasi file label YOLO."""
        errors = []
        stats = {
            'class_ids': set(),
            'bbox_count': 0,
            'image_size': None
        }
        
        if not label_path or not label_path.exists():
            errors.append(f"File label tidak ditemukan untuk {img_path.name}")
            return False, errors, stats
        
        try:
            # Baca ukuran gambar
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Tidak dapat membaca gambar untuk validasi label: {img_path.name}")
                return False, errors, stats
            
            img_h, img_w = img.shape[:2]
            stats['image_size'] = (img_w, img_h)
            
            # Baca file label
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                errors.append(f"File label kosong: {label_path.name}")
                return False, errors, stats
            
            # Validasi setiap baris
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 5:  # class_id, x_center, y_center, width, height
                        errors.append(f"Format tidak valid di {label_path.name} baris {i}: '{line}'")
                        continue
                    
                    # Validasi class_id
                    class_id = int(parts[0])
                    if class_id < 0 or (self.required_classes and class_id >= len(self.required_classes)):
                        errors.append(
                            f"ID kelas tidak valid di {label_path.name} baris {i}: {class_id} "
                            f"(harus antara 0-{len(self.required_classes)-1})"
                        )
                    
                    # Validasi koordinat
                    x_center, y_center = float(parts[1]), float(parts[2])
                    width, height = float(parts[3]), float(parts[4])
                    
                    if not (0 <= x_center <= 1) or not (0 <= y_center <= 1) or \
                       not (0 < width <= 1) or not (0 < height <= 1):
                        errors.append(
                            f"Koordinat tidak valid di {label_path.name} baris {i}: "
                            f"x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f} "
                            "(harus dalam rentang [0,1] untuk x,y dan (0,1] untuk w,h)"
                        )
                    
                    # Update statistik
                    stats['class_ids'].add(class_id)
                    stats['bbox_count'] += 1
                    
                except (ValueError, IndexError) as e:
                    errors.append(f"Error parsing {label_path.name} baris {i}: {str(e)}")
            
            return len(errors) == 0, errors, stats
            
        except Exception as e:
            errors.append(f"Error saat memvalidasi {label_path.name}: {str(e)}")
            return False, errors, stats
    
    def _get_corresponding_label_file(self, img_path: Path, label_dir: Path) -> Optional[Path]:
        """Dapatkan path ke file label yang sesuai dengan file gambar."""
        label_file = label_dir / f"{img_path.stem}.txt"
        return label_file if label_file.exists() else None


def create_validation_engine(config: Dict[str, Any]) -> ValidationEngine:
    """Factory function untuk membuat instance ValidationEngine."""
    return ValidationEngine(config)

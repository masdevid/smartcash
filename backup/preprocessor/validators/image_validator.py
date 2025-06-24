"""
Validator untuk file gambar dalam dataset YOLOv5.

Modul ini menyediakan fungsionalitas untuk memvalidasi file gambar
sebelum diproses lebih lanjut dalam pipeline preprocessing.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from smartcash.common.logger import get_logger


class ImageValidator:
    """Validator untuk memeriksa integritas dan kesesuaian gambar."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inisialisasi ImageValidator dengan konfigurasi.
        
        Args:
            config: Konfigurasi validator (opsional)
        """
        self.config = config or {}
        self.logger = get_logger()
        
        # Konfigurasi default
        self.min_size = self.config.get('min_image_size', 32)
        self.max_size = self.config.get('max_image_size', 4096)
        self.max_file_size = self.config.get('max_file_size_mb', 10)  # MB
        self.allowed_formats = self.config.get('allowed_formats', ['.jpg', '.jpeg', '.png'])
    
    def validate(self, image_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validasi file gambar.
        
        Args:
            image_path: Path ke file gambar yang akan divalidasi
            
        Returns:
            Tuple berisi:
                - bool: True jika valid, False jika tidak
                - List[str]: Daftar pesan error (jika ada)
                - Dict[str, Any]: Statistik validasi
        """
        errors = []
        stats = {
            'file_size_mb': 0,
            'width': 0,
            'height': 0,
            'channels': 0,
            'format': '',
            'is_corrupt': False
        }
        
        try:
            # Periksa ekstensi file
            if image_path.suffix.lower() not in self.allowed_formats:
                errors.append(
                    f"Format file tidak didukung: {image_path.suffix}. "
                    f"Format yang didukung: {', '.join(self.allowed_formats)}"
                )
                return False, errors, stats
            
            # Periksa ukuran file
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)  # Convert ke MB
            stats['file_size_mb'] = round(file_size_mb, 2)
            
            if file_size_mb > self.max_file_size:
                errors.append(
                    f"Ukuran file terlalu besar: {file_size_mb:.2f}MB "
                    f"(maksimum: {self.max_file_size}MB)"
                )
            
            # Baca gambar
            img = cv2.imread(str(image_path))
            if img is None:
                errors.append("Gagal membaca file gambar (mungkin korup atau format tidak valid)")
                stats['is_corrupt'] = True
                return False, errors, stats
            
            # Dapatkan properti gambar
            h, w = img.shape[:2]
            channels = 1 if len(img.shape) == 2 else img.shape[2]
            
            # Update statistik
            stats.update({
                'width': w,
                'height': h,
                'channels': channels,
                'format': image_path.suffix.lower(),
                'aspect_ratio': round(w / h, 2) if h > 0 else 0
            })
            
            # Validasi ukuran
            if w < self.min_size or h < self.min_size:
                errors.append(
                    f"Gambar terlalu kecil: {w}x{h} (minimum: {self.min_size}x{self.min_size})"
                )
            
            if w > self.max_size or h > self.max_size:
                errors.append(
                    f"Gambar terlalu besar: {w}x{h} (maksimum: {self.max_size}x{self.max_size})"
                )
            
            # Validasi rasio aspek ekstrim
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                errors.append(
                    f"Rasio aspek ekstrim: {aspect_ratio:.2f} "
                    "(mungkin ada masalah dengan orientasi gambar)"
                )
            
            # Periksa kualitas gambar (blurriness)
            if channels == 3:  # Hanya untuk gambar berwarna
                blur_value = self._calculate_blur_metric(img)
                stats['blur_metric'] = round(blur_value, 4)
                
                if blur_value < 100:  # Nilai ambang untuk blur
                    errors.append("Gambar terdeteksi blur atau tidak fokus")
            
            return len(errors) == 0, errors, stats
            
        except Exception as e:
            error_msg = f"Error saat memvalidasi {image_path.name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return False, errors, stats
    
    @staticmethod
    def _calculate_blur_metric(image: np.ndarray) -> float:
        """Hitung metrik blur menggunakan varians Laplacian.
        
        Args:
            image: Gambar dalam format BGR (OpenCV)
            
        Returns:
            Nilai varians Laplacian sebagai indikator ketajaman
        """
        try:
            # Konversi ke grayscale jika perlu
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Hitung varians Laplacian
            return cv2.Laplacian(gray, cv2.CV_64F).var()
            
        except Exception:
            return 0.0
    
    def batch_validate(self, image_paths: List[Path]) -> Dict[Path, Dict[str, Any]]:
        """Validasi sekumpulan file gambar.
        
        Args:
            image_paths: Daftar path ke file gambar yang akan divalidasi
            
        Returns:
            Dictionary dengan path gambar sebagai key dan dictionary berisi:
            - valid: bool
            - errors: List[str]
            - stats: Dict[str, Any]
        """
        results = {}
        
        for img_path in image_paths:
            is_valid, errors, stats = self.validate(img_path)
            results[img_path] = {
                'valid': is_valid,
                'errors': errors,
                'stats': stats
            }
        
        return results


def create_image_validator(config: Dict[str, Any] = None) -> ImageValidator:
    """Factory function untuk membuat instance ImageValidator."""
    return ImageValidator(config)

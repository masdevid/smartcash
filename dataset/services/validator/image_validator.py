"""
File: smartcash/dataset/services/validator/image_validator.py
Deskripsi: Implementasi validator untuk gambar dalam dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from smartcash.common.logger import get_logger


class ImageValidator:
    """Validator khusus untuk memeriksa dan memvalidasi gambar dataset."""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi ImageValidator.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("image_validator")
        
        # Setup parameter validasi
        validation_config = self.config.get('validation', {}).get('image', {})
        self.min_width = validation_config.get('min_width', 100)
        self.min_height = validation_config.get('min_height', 100)
        self.min_contrast = validation_config.get('min_contrast', 20)
        self.min_sharpness = validation_config.get('min_sharpness', 100)
        self.max_file_size_mb = validation_config.get('max_file_size_mb', 10)
        self.supported_formats = validation_config.get('supported_formats', ['.jpg', '.jpeg', '.png'])
    
    def validate_image(self, image_path: Path) -> Tuple[bool, List[str]]:
        """
        Validasi satu file gambar.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Tuple (valid, list_masalah)
        """
        issues = []
        
        # Cek eksistensi file
        if not image_path.exists():
            issues.append("File tidak ditemukan")
            return False, issues
            
        # Cek ekstensi file
        if image_path.suffix.lower() not in self.supported_formats:
            issues.append(f"Format file tidak didukung: {image_path.suffix}")
            return False, issues
            
        # Cek ukuran file
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            issues.append(f"Ukuran file terlalu besar: {file_size_mb:.2f}MB (maks: {self.max_file_size_mb}MB)")
        
        # Coba baca gambar
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                issues.append("Gambar tidak dapat dibaca")
                return False, issues
                
            # Cek dimensi
            h, w = img.shape[:2]
            if w < self.min_width or h < self.min_height:
                issues.append(f"Dimensi gambar terlalu kecil: {w}x{h} (min: {self.min_width}x{self.min_height})")
            
            # Cek kontras
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            if contrast < self.min_contrast:
                issues.append(f"Kontras gambar terlalu rendah: {contrast:.2f} (min: {self.min_contrast})")
            
            # Cek ketajaman (blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.min_sharpness:
                issues.append(f"Gambar terlalu blur: {laplacian_var:.2f} (min: {self.min_sharpness})")
            
            # Cek aspek rasio ekstrim
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                issues.append(f"Aspek rasio tidak normal: {aspect_ratio:.2f}")
            
            # Cek channels (harus RGB)
            if len(img.shape) < 3 or img.shape[2] != 3:
                issues.append(f"Format channel tidak valid: {img.shape}")
                
        except Exception as e:
            issues.append(f"Error saat memproses gambar: {str(e)}")
            return False, issues
            
        # Gambar valid jika tidak ada masalah
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def fix_image(self, image_path: Path) -> Tuple[bool, List[str]]:
        """
        Perbaiki masalah pada gambar.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Tuple (berhasil_diperbaiki, perbaikan_yang_dilakukan)
        """
        fixes = []
        
        try:
            # Baca gambar
            img = cv2.imread(str(image_path))
            if img is None:
                return False, ["Gambar tidak dapat dibaca"]
            
            # Cek dan perbaiki dimensi
            h, w = img.shape[:2]
            if w < self.min_width or h < self.min_height:
                # Resize gambar
                new_w = max(w, self.min_width)
                new_h = max(h, self.min_height)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                fixes.append(f"Resize gambar {w}x{h} → {new_w}x{new_h}")
            
            # Perbaiki kontras
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            if contrast < self.min_contrast:
                # Gunakan CLAHE untuk perbaiki kontras
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Terapkan ke semua channel
                if len(img.shape) == 3:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    cl = clahe.apply(l)
                    merged = cv2.merge((cl, a, b))
                    img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                else:
                    img = enhanced
                
                fixes.append(f"Perbaikan kontras {contrast:.2f} → {cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std():.2f}")
            
            # Perbaiki blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.min_sharpness:
                # Sharpening dengan kernel
                kernel = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
                img = cv2.filter2D(img, -1, kernel)
                
                # Ukur ketajaman setelah perbaikan
                new_sharpness = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                fixes.append(f"Perbaikan ketajaman {laplacian_var:.2f} → {new_sharpness:.2f}")
            
            # Simpan hasil perbaikan
            if fixes:
                cv2.imwrite(str(image_path), img)
                self.logger.info(f"🔧 Gambar {image_path.name} diperbaiki: {', '.join(fixes)}")
                return True, fixes
                
            return False, []
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memperbaiki gambar {image_path.name}: {str(e)}")
            return False, [f"Error: {str(e)}"]
    
    def get_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Dapatkan metadata gambar.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Dictionary berisi metadata gambar
        """
        metadata = {
            'path': str(image_path),
            'filename': image_path.name,
            'format': image_path.suffix,
            'size_bytes': 0,
            'size_mb': 0,
            'width': 0,
            'height': 0,
            'channels': 0,
            'is_valid': False
        }
        
        if not image_path.exists():
            return metadata
            
        # Tambahkan info ukuran file
        file_size = os.path.getsize(image_path)
        metadata['size_bytes'] = file_size
        metadata['size_mb'] = file_size / (1024 * 1024)
        
        # Coba baca gambar untuk info dimensi
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
                
                # Update metadata
                metadata.update({
                    'width': w,
                    'height': h,
                    'channels': channels,
                    'aspect_ratio': w / h if h > 0 else 0,
                    'contrast': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std(),
                    'sharpness': cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
                    'is_valid': True
                })
                
        except Exception:
            pass
            
        return metadata
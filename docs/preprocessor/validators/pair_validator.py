"""
Validator untuk memeriksa konsistensi pasangan gambar dan label.

Modul ini menyediakan fungsionalitas untuk memvalidasi konsistensi antara
file gambar dan file label yang terkait dalam dataset YOLO.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set

from smartcash.common.logger import get_logger


class PairValidator:
    """Validator untuk memeriksa konsistensi pasangan gambar dan label."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inisialisasi PairValidator dengan konfigurasi.
        
        Args:
            config: Konfigurasi validator (opsional)
        """
        self.config = config or {}
        self.logger = get_logger()
        
        # Konfigurasi default
        self.require_label = self.config.get('require_label', True)
        self.require_image = self.config.get('require_image', True)
        self.allowed_image_exts = self.config.get('allowed_image_exts', ['.jpg', '.jpeg', '.png'])
        self.label_ext = self.config.get('label_extension', '.txt')
    
    def validate_pair(self, image_path: Path, label_path: Optional[Path] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validasi pasangan gambar dan label.
        
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (opsional, akan dihitung otomatis jika None)
            
        Returns:
            Tuple berisi:
                - bool: True jika valid, False jika tidak
                - List[str]: Daftar pesan error (jika ada)
                - Dict[str, Any]: Statistik validasi
        """
        errors = []
        stats = {
            'image_exists': False,
            'label_exists': False,
            'is_paired': False,
            'has_orphan': False
        }
        
        try:
            # Validasi path gambar
            if not image_path.exists():
                if self.require_image:
                    errors.append(f"File gambar tidak ditemukan: {image_path}")
                return False, errors, stats
            
            stats['image_exists'] = True
            
            # Jika label_path tidak disediakan, cari berdasarkan nama file gambar
            if label_path is None:
                label_path = self._get_label_path(image_path)
            
            # Validasi path label
            if not label_path.exists():
                if self.require_label:
                    errors.append(f"File label tidak ditemukan: {label_path}")
                    stats['has_orphan'] = True
                return len(errors) == 0, errors, stats
            
            stats['label_exists'] = True
            
            # Periksa apakah label kosong
            if label_path.stat().st_size == 0:
                errors.append(f"File label kosong: {label_path}")
                return False, errors, stats
            
            # Periksa konsistensi nama file (tanpa ekstensi harus sama)
            if image_path.stem != label_path.stem:
                errors.append(
                    f"Nama file gambar dan label tidak konsisten: "
                    f"{image_path.name} vs {label_path.name}"
                )
            
            # Jika semua validasi berhasil
            if not errors:
                stats['is_paired'] = True
            
            return len(errors) == 0, errors, stats
            
        except Exception as e:
            error_msg = f"Error saat memvalidasi pasangan {image_path.name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return False, errors, stats
    
    def validate_pairs(self, image_paths: List[Path], label_paths: Optional[List[Path]] = None) -> Dict[Path, Dict[str, Any]]:
        """Validasi sekumpulan pasangan gambar dan label.
        
        Args:
            image_paths: Daftar path ke file gambar
            label_paths: Daftar path ke file label (opsional, akan dihitung otomatis jika None)
            
        Returns:
            Dictionary dengan path gambar sebagai key dan dictionary berisi:
            - valid: bool
            - errors: List[str]
            - stats: Dict[str, Any]
        """
        results = {}
        
        if label_paths is not None and len(image_paths) != len(label_paths):
            self.logger.warning("Jumlah image_paths dan label_paths tidak sama, menggunakan pencarian otomatis untuk label")
            label_paths = None
        
        for i, img_path in enumerate(image_paths):
            lbl_path = label_paths[i] if label_paths else None
            is_valid, errors, stats = self.validate_pair(img_path, lbl_path)
            results[img_path] = {
                'valid': is_valid,
                'errors': errors,
                'stats': stats
            }
        
        return results
    
    def find_orphans(self, image_dir: Path, label_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
        """Temukan file gambar/label yang tidak memiliki pasangan.
        
        Args:
            image_dir: Direktori berisi file gambar
            label_dir: Direktori berisi file label (opsional, default: sama dengan image_dir)
            
        Returns:
            Dictionary dengan dua key:
            - 'orphan_images': List[Path] - Gambar tanpa label
            - 'orphan_labels': List[Path] - Label tanpa gambar
        """
        if label_dir is None:
            label_dir = image_dir
        
        # Dapatkan daftar file gambar dan label
        image_files = self._find_files(image_dir, self.allowed_image_exts)
        label_files = self._find_files(label_dir, [self.label_ext])
        
        # Buat set nama file tanpa ekstensi untuk pencarian cepat
        image_names = {f.stem for f in image_files}
        label_names = {f.stem for f in label_files}
        
        # Temukan file yang tidak memiliki pasangan
        orphan_images = [f for f in image_files if f.stem not in label_names]
        orphan_labels = [f for f in label_files if f.stem not in image_names]
        
        return {
            'orphan_images': orphan_images,
            'orphan_labels': orphan_labels
        }
    
    def _get_label_path(self, image_path: Path) -> Path:
        """Dapatkan path label yang sesuai untuk file gambar."""
        return image_path.with_suffix(self.label_ext)
    
    @staticmethod
    def _find_files(directory: Path, extensions: List[str]) -> List[Path]:
        """Temukan semua file dalam direktori dengan ekstensi yang sesuai."""
        if not directory.exists() or not directory.is_dir():
            return []
        
        files = []
        for ext in extensions:
            files.extend(list(directory.glob(f'*{ext}')))
        
        return files


def create_pair_validator(config: Dict[str, Any] = None) -> PairValidator:
    """Factory function untuk membuat instance PairValidator."""
    return PairValidator(config)

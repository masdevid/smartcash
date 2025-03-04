# File: smartcash/utils/path_validator.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk validasi dan koreksi jalur data

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from smartcash.utils.logger import SmartCashLogger

class PathValidator:
    """
    Utilitas untuk memvalidasi dan memperbaiki jalur-jalur dataset.
    Membantu mendeteksi dan menangani perbedaan seperti 'val' vs 'valid'.
    """
    
    # Mapping variant paths yang umum ditemui
    PATH_VARIANTS = {
        'train': ['training', 'tr'],
        'valid': ['val', 'validation', 'dev'],
        'test': ['testing', 'te', 'eval']
    }
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi validator jalur.
        
        Args:
            logger: Logger opsional
        """
        self.logger = logger or SmartCashLogger(__name__)
    
    def validate_data_structure(self, data_dir: Union[str, Path]) -> Dict[str, bool]:
        """
        Validasi struktur direktori data lengkap.
        
        Args:
            data_dir: Direktori data utama
            
        Returns:
            Dict hasil validasi untuk setiap split
        """
        data_path = Path(data_dir)
        results = {}
        
        # Validasi setiap split
        for split in ['train', 'valid', 'test']:
            # Periksa folder standar
            standard_path = data_path / split
            found = self._validate_split_dir(standard_path)
            
            # Jika tidak ditemukan, periksa varian
            if not found:
                for variant in self.PATH_VARIANTS.get(split, []):
                    variant_path = data_path / variant
                    if self._validate_split_dir(variant_path):
                        self.logger.info(f"🔄 Menggunakan varian jalur untuk {split}: {variant_path}")
                        found = True
                        break
            
            results[split] = found
            
        return results
    
    def _validate_split_dir(self, split_path: Path) -> bool:
        """
        Validasi satu direktori split.
        
        Args:
            split_path: Path ke direktori split
            
        Returns:
            Boolean yang menunjukkan kevalidan
        """
        # Periksa struktur dasar: harus ada direktori images dan labels
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        if not split_path.exists():
            return False
            
        dirs_exist = images_dir.exists() and labels_dir.exists()
        
        if dirs_exist:
            # Periksa apakah direktori-direktori tersebut memiliki konten
            has_images = any(images_dir.glob('*.*'))
            has_labels = any(labels_dir.glob('*.*'))
            
            if not has_images:
                self.logger.warning(f"⚠️ Direktori {images_dir} ada tapi kosong")
            if not has_labels:
                self.logger.warning(f"⚠️ Direktori {labels_dir} ada tapi kosong")
                
            return has_images and has_labels
        
        return False
    
    def find_or_fix_path(
        self, 
        base_path: Union[str, Path], 
        target: str,
        auto_fix: bool = False
    ) -> Tuple[Path, bool]:
        """
        Cari jalur yang benar atau perbaiki jalur yang salah.
        
        Args:
            base_path: Jalur dasar
            target: Target split ('train', 'valid', 'test')
            auto_fix: Buat symlink otomatis jika jalur yang benar ditemukan
            
        Returns:
            Tuple (jalur yang terkoreksi, bool untuk menunjukkan apakah jalur ditemukan)
        """
        base = Path(base_path)
        
        # Cek jalur standar
        standard_path = base / target
        if standard_path.exists() and (standard_path / 'images').exists():
            return standard_path, True
        
        # Cek jalur varian
        if target in self.PATH_VARIANTS:
            for variant in self.PATH_VARIANTS[target]:
                variant_path = base / variant
                if variant_path.exists() and (variant_path / 'images').exists():
                    self.logger.info(f"🔍 Menemukan varian untuk {target}: {variant_path}")
                    
                    # Jika auto_fix aktif, buat symlink
                    if auto_fix:
                        try:
                            # Buat direktori target jika belum ada
                            standard_path.mkdir(exist_ok=True, parents=True)
                            
                            # Buat symlink untuk images
                            images_link = standard_path / 'images'
                            if not images_link.exists():
                                os.symlink(variant_path / 'images', images_link)
                                
                            # Buat symlink untuk labels
                            labels_link = standard_path / 'labels'
                            if not labels_link.exists() and (variant_path / 'labels').exists():
                                os.symlink(variant_path / 'labels', labels_link)
                                
                            self.logger.success(
                                f"✅ Berhasil membuat symlink: {target} ➡️ {variant}"
                            )
                            
                            return standard_path, True
                            
                        except Exception as e:
                            self.logger.error(f"❌ Gagal membuat symlink: {str(e)}")
                    
                    # Jika tidak auto_fix, kembalikan varian yang ditemukan
                    return variant_path, True
        
        # Jika tidak ditemukan jalur yang valid
        return standard_path, False
    
    def suggest_fixes(self, data_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Sarankan perbaikan untuk struktur direktori data.
        
        Args:
            data_dir: Direktori data utama
            
        Returns:
            Dict berisi saran perbaikan
        """
        data_path = Path(data_dir)
        results = self.validate_data_structure(data_path)
        suggestions = {}
        
        for split, valid in results.items():
            if not valid:
                # Cek apakah ada varian yang valid
                found_variant = False
                
                for variant in self.PATH_VARIANTS.get(split, []):
                    variant_path = data_path / variant
                    if self._validate_split_dir(variant_path):
                        suggestions[split] = f"Gunakan symlink: ln -s {variant} {split}"
                        found_variant = True
                        break
                
                if not found_variant:
                    suggestions[split] = f"Buat direktori {split}/images dan {split}/labels"
        
        return suggestions
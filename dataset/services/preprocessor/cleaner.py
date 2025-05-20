"""
File: smartcash/dataset/services/preprocessor/cleaner.py
Deskripsi: Pembersih cache dataset preprocessed untuk menghemat ruang penyimpanan
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time

from smartcash.common.logger import get_logger
from smartcash.dataset.services.preprocessor.storage import PreprocessedStorage


class PreprocessedCleaner:
    """
    Pembersih cache dataset preprocessed.
    Membantu mengelola ruang penyimpanan dengan membersihkan hasil preprocessing
    yang sudah tidak dibutuhkan atau sudah kadaluarsa.
    """
    
    def __init__(
        self,
        preprocessed_dir: Union[str, Path] = "data/preprocessed",
        max_age_days: int = 30,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi pembersih preprocessed.
        
        Args:
            preprocessed_dir: Direktori data preprocessed
            max_age_days: Usia maksimum data preprocessed (dalam hari)
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_age_days = max_age_days
        
        # Inisialisasi storage manager untuk metadata
        self.storage = PreprocessedStorage(self.preprocessed_dir, logger=self.logger)
        
        self.logger.info(f"🧹 PreprocessedCleaner diinisialisasi (dir: {preprocessed_dir}, max_age: {max_age_days} hari)")
    
    def clean_expired(self) -> Dict[str, int]:
        """
        Bersihkan data preprocessed yang sudah kadaluarsa.
        
        Returns:
            Dictionary statistik pembersihan
        """
        now = time.time()
        max_age_seconds = self.max_age_days * 24 * 60 * 60
        
        cleaned_stats = {
            'total_cleaned': 0,
            'splits_cleaned': 0,
            'bytes_freed': 0
        }
        
        # Baca metadata untuk mendapatkan waktu pembuatan setiap split
        metadata = self.storage._load_metadata()
        splits_metadata = metadata.get('splits', {})
        
        for split, split_meta in splits_metadata.items():
            # Cek apakah split sudah kadaluarsa
            created_time = split_meta.get('created_time', 0)
            if now - created_time > max_age_seconds:
                split_dir = self.preprocessed_dir / split
                
                # Hitung ukuran sebelum dibersihkan
                split_size = self._get_directory_size(split_dir)
                
                # Bersihkan split
                self.logger.info(f"🧹 Membersihkan split {split} (umur: {(now - created_time) / (24*60*60):.1f} hari)")
                
                # Hapus direktori
                self.storage.clean_storage(split)
                
                # Update statistik
                cleaned_stats['total_cleaned'] += 1
                cleaned_stats['splits_cleaned'] += 1
                cleaned_stats['bytes_freed'] += split_size
        
        self.logger.success(
            f"✅ Pembersihan selesai: {cleaned_stats['splits_cleaned']} split, "
            f"{cleaned_stats['bytes_freed'] / (1024*1024):.2f} MB dibebaskan"
        )
        
        return cleaned_stats
    
    def clean_all(self) -> Dict[str, int]:
        """
        Bersihkan semua data preprocessed.
        
        Returns:
            Dictionary statistik pembersihan
        """
        # Hitung ukuran sebelum dibersihkan
        total_size = self._get_directory_size(self.preprocessed_dir)
        
        # Bersihkan semua
        self.storage.clean_storage()
        
        cleaned_stats = {
            'total_cleaned': 1,
            'splits_cleaned': 3,  # Asumsi 3 split standar
            'bytes_freed': total_size
        }
        
        self.logger.success(
            f"✅ Pembersihan lengkap selesai: "
            f"{cleaned_stats['bytes_freed'] / (1024*1024):.2f} MB dibebaskan"
        )
        
        return cleaned_stats
    
    def clean_split(self, split: str) -> Dict[str, int]:
        """
        Bersihkan data preprocessed untuk split tertentu.
        
        Args:
            split: Nama split yang akan dibersihkan
            
        Returns:
            Dictionary statistik pembersihan
        """
        split_dir = self.preprocessed_dir / split
        
        # Cek apakah split ada
        if not split_dir.exists():
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan")
            return {
                'total_cleaned': 0,
                'splits_cleaned': 0,
                'bytes_freed': 0
            }
        
        # Hitung ukuran sebelum dibersihkan
        split_size = self._get_directory_size(split_dir)
        
        # Bersihkan split
        self.storage.clean_storage(split)
        
        cleaned_stats = {
            'total_cleaned': 1,
            'splits_cleaned': 1,
            'bytes_freed': split_size
        }
        
        self.logger.success(
            f"✅ Pembersihan split {split} selesai: "
            f"{cleaned_stats['bytes_freed'] / (1024*1024):.2f} MB dibebaskan"
        )
        
        return cleaned_stats
    
    def _get_directory_size(self, directory: Union[str, Path]) -> int:
        """
        Hitung ukuran direktori dalam bytes.
        
        Args:
            directory: Path direktori
            
        Returns:
            Ukuran dalam bytes
        """
        total_size = 0
        directory = Path(directory)
        
        if not directory.exists():
            return 0
            
        for path in directory.glob('**/*'):
            if path.is_file():
                total_size += path.stat().st_size
                
        return total_size
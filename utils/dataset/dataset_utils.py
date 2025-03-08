"""
File: smartcash/utils/dataset/dataset_utils.py
Author: Alfrida Sabar
Deskripsi: Utilitas untuk operasi umum pada dataset seperti pencarian file, backup, dan manipulasi direktori
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger

class DatasetUtils:
    """
    Utilitas untuk operasi umum pada dataset.
    """
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi dataset utils.
        
        Args:
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger(__name__)
    
    def find_image_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Temukan semua file gambar dalam direktori.
        
        Args:
            directory: Direktori yang akan dicari
            
        Returns:
            List path file gambar
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self.logger.warning(f"⚠️ Direktori tidak ditemukan: {dir_path}")
            return []
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(dir_path.glob(ext)))
            
        if not image_files:
            self.logger.info(f"ℹ️ Tidak ada gambar ditemukan di {dir_path}")
            
        return image_files
    
    def get_random_sample(self, items: List, sample_size: int) -> List:
        """
        Ambil sampel acak dari list.
        
        Args:
            items: List item
            sample_size: Ukuran sampel
            
        Returns:
            Sampel acak dari items
        """
        if sample_size <= 0 or sample_size >= len(items):
            return items
            
        random.seed(42)  # Untuk hasil yang konsisten
        return random.sample(items, sample_size)
    
    def backup_directory(self, source_dir: Union[str, Path], suffix: Optional[str] = None) -> Optional[Path]:
        """
        Buat backup direktori.
        
        Args:
            source_dir: Direktori sumber
            suffix: Suffix untuk nama direktori backup (default: timestamp)
            
        Returns:
            Path direktori backup atau None jika gagal
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            self.logger.warning(f"⚠️ Direktori sumber tidak ditemukan: {source_path}")
            return None
        
        # Tentukan suffix jika tidak disediakan
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Buat path backup
        parent_dir = source_path.parent
        backup_path = parent_dir / f"{source_path.name}_backup_{suffix}"
        
        try:
            # Jika backup sudah ada, tambahkan angka
            if backup_path.exists():
                i = 1
                while backup_path.exists():
                    backup_path = parent_dir / f"{source_path.name}_backup_{suffix}_{i}"
                    i += 1
            
            # Salin direktori
            shutil.copytree(source_path, backup_path)
            
            self.logger.success(f"✅ Direktori berhasil dibackup ke: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backup {source_path}: {str(e)}")
            return None
    
    def move_invalid_files(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        file_list: List[Path]
    ) -> Dict[str, int]:
        """
        Pindahkan file ke direktori target.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            file_list: List file yang akan dipindahkan
            
        Returns:
            Dict statistik pemindahan
        """
        stats = {'moved': 0, 'skipped': 0, 'errors': 0}
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in file_list:
            try:
                # Buat target path yang sama dengan struktur source
                rel_path = file_path.relative_to(source_dir)
                dest_path = target_path / rel_path
                
                # Pastikan direktori tujuan ada
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Salin file jika belum ada
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    stats['moved'] += 1
                else:
                    stats['skipped'] += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Gagal memindahkan {file_path}: {str(e)}")
                stats['errors'] += 1
        
        return stats
    
    def create_directory_structure(self, base_dir: Union[str, Path], subdirs: List[str]) -> Dict[str, int]:
        """
        Buat struktur direktori.
        
        Args:
            base_dir: Direktori utama
            subdirs: List subdirektori yang akan dibuat
            
        Returns:
            Dict statistik pembuatan direktori
        """
        stats = {'created': 0, 'existing': 0, 'errors': 0}
        base_path = Path(base_dir)
        
        for subdir in subdirs:
            try:
                dir_path = base_path / subdir
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    stats['created'] += 1
                else:
                    stats['existing'] += 1
            except Exception as e:
                self.logger.error(f"❌ Gagal membuat direktori {subdir}: {str(e)}")
                stats['errors'] += 1
                
        return stats
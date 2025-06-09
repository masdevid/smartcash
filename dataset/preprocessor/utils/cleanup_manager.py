"""
File: smartcash/dataset/preprocessor/utils/cleanup_manager.py
Deskripsi: Modul untuk menangani pembersihan file dan direktori
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from smartcash.common.logger import get_logger

class CleanupManager:
    """Kelas untuk menangani pembersihan file dan direktori"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inisialisasi CleanupManager
        
        Args:
            config: Konfigurasi cleanup
        """
        self.config = config.get('preprocessing', {}).get('cleanup', {})
        self.logger = get_logger()
        self.files_to_cleanup: Set[Path] = set()
        self.dirs_to_cleanup: Set[Path] = set()
    
    def register_file(self, file_path: Path) -> None:
        """Daftarkan file untuk dibersihkan nanti
        
        Args:
            file_path: Path ke file yang akan didaftarkan
        """
        if file_path and file_path.is_file():
            self.files_to_cleanup.add(file_path)
    
    def register_dir(self, dir_path: Path, recursive: bool = False) -> None:
        """Daftarkan direktori untuk dibersihkan nanti
        
        Args:
            dir_path: Path ke direktori yang akan didaftarkan
            recursive: Jika True, daftarkan semua isi direktori secara rekursif
        """
        if dir_path and dir_path.is_dir():
            self.dirs_to_cleanup.add((dir_path, recursive))
    
    def cleanup(self) -> Dict[str, int]:
        """Bersihkan semua file dan direktori yang terdaftar
        
        Returns:
            Dict berisi statistik cleanup
        """
        stats = {
            'files_deleted': 0,
            'files_failed': 0,
            'dirs_deleted': 0,
            'dirs_failed': 0
        }
        
        # Hapus file-file yang terdaftar
        for file_path in list(self.files_to_cleanup):
            try:
                if file_path.exists():
                    file_path.unlink()
                    stats['files_deleted'] += 1
                self.files_to_cleanup.remove(file_path)
            except Exception as e:
                self.logger.error(f"Gagal menghapus file {file_path}: {str(e)}")
                stats['files_failed'] += 1
        
        # Hapus direktori yang terdaftar
        for dir_path, recursive in list(self.dirs_to_cleanup):
            try:
                if dir_path.exists():
                    if recursive:
                        shutil.rmtree(dir_path)
                    else:
                        dir_path.rmdir()
                    stats['dirs_deleted'] += 1
                self.dirs_to_cleanup.remove((dir_path, recursive))
            except Exception as e:
                self.logger.error(f"Gagal menghapus direktori {dir_path}: {str(e)}")
                stats['dirs_failed'] += 1
        
        return stats
    
    def cleanup_output_dirs(self, split: Optional[str] = None) -> Dict[str, int]:
        """Bersihkan direktori output
        
        Args:
            split: Nama split yang akan dibersihkan (None untuk semua split)
            
        Returns:
            Dict berisi statistik cleanup
        """
        from .path_resolver import PathResolver
        
        resolver = PathResolver(self.config)
        stats = {
            'files_deleted': 0,
            'dirs_deleted': 0
        }
        
        # Tentukan splits yang akan dibersihkan
        splits = [split] if split else ['train', 'val', 'test']
        
        for current_split in splits:
            # Hapus direktori gambar yang telah diproses
            img_dir = resolver.get_preprocessed_image_dir(current_split)
            if img_dir.exists():
                try:
                    shutil.rmtree(img_dir)
                    stats['dirs_deleted'] += 1
                except Exception as e:
                    self.logger.error(f"Gagal membersihkan direktori gambar {img_dir}: {str(e)}")
            
            # Hapus direktori label yang telah diproses
            label_dir = resolver.get_preprocessed_label_dir(current_split)
            if label_dir.exists():
                try:
                    shutil.rmtree(label_dir)
                    stats['dirs_deleted'] += 1
                except Exception as e:
                    self.logger.error(f"Gagal membersihkan direktori label {label_dir}: {str(e)}")
        
        # Hapus file-file temporary
        temp_dir = resolver.get_temp_dir()
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)  # Buat kembali direktori temp
                stats['dirs_deleted'] += 1
            except Exception as e:
                self.logger.error(f"Gagal membersihkan direktori temporary {temp_dir}: {str(e)}")
        
        return stats
    
    def __del__(self):
        """Destruktor: Bersihkan file dan direktori yang terdaftar"""
        if self.files_to_cleanup or self.dirs_to_cleanup:
            self.cleanup()

"""
File: smartcash/dataset/preprocessor/utils/path_resolver.py
Deskripsi: Modul untuk menangani resolusi path file dan direktori
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

class PathResolver:
    """Kelas untuk menangani resolusi path file dan direktori"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inisialisasi PathResolver
        
        Args:
            config: Konfigurasi yang berisi path dasar
        """
        self.config = config
        self.data_dir = Path(config.get('data', {}).get('dir', 'data'))
        self.raw_data_dir = self.data_dir / 'raw'
        self.preprocessed_dir = Path(config.get('data', {}).get('output', {}).get('preprocessed', 'data/preprocessed'))
        
        # Pastikan direktori dasar ada
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    def get_source_image_dir(self, split: str) -> Path:
        """Dapatkan direktori sumber gambar untuk split tertentu
        
        Args:
            split: Nama split ('train', 'val', 'test')
            
        Returns:
            Path ke direktori gambar sumber
        """
        # Coba dapatkan dari konfigurasi terlebih dahulu
        split_path = self.config.get('data', {}).get('splits', {}).get(split)
        if split_path:
            return Path(split_path) / 'images'
            
        # Fallback ke struktur direktori default
        return self.raw_data_dir / split / 'images'
    
    def get_source_label_dir(self, split: str) -> Path:
        """Dapatkan direktori sumber label untuk split tertentu
        
        Args:
            split: Nama split ('train', 'val', 'test')
            
        Returns:
            Path ke direktori label sumber
        """
        # Coba dapatkan dari konfigurasi terlebih dahulu
        split_path = self.config.get('data', {}).get('splits', {}).get(split)
        if split_path:
            return Path(split_path) / 'labels'
            
        # Fallback ke struktur direktori default
        return self.raw_data_dir / split / 'labels'
    
    def get_preprocessed_image_dir(self, split: str) -> Path:
        """Dapatkan direktori tujuan gambar yang telah diproses
        
        Args:
            split: Nama split ('train', 'val', 'test')
            
        Returns:
            Path ke direktori gambar yang telah diproses
        """
        if self.config.get('preprocessing', {}).get('output', {}).get('organize_by_split', True):
            return self.preprocessed_dir / split / 'images'
        return self.preprocessed_dir / 'images'
    
    def get_preprocessed_label_dir(self, split: str) -> Path:
        """Dapatkan direktori tujuan label yang telah diproses
        
        Args:
            split: Nama split ('train', 'val', 'test')
            
        Returns:
            Path ke direktori label yang telah diproses
        """
        if self.config.get('preprocessing', {}).get('output', {}).get('organize_by_split', True):
            return self.preprocessed_dir / split / 'labels'
        return self.preprocessed_dir / 'labels'
    
    def get_relative_path(self, path: Path, base_dir: Path) -> str:
        """Dapatkan path relatif dari direktori dasar
        
        Args:
            path: Path lengkap ke file
            base_dir: Direktori dasar
            
        Returns:
            String path relatif
        """
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)
    
    def resolve_output_path(self, input_path: Path, input_base: Path, output_base: Path) -> Path:
        """Resolve path output berdasarkan path input
        
        Args:
            input_path: Path input lengkap
            input_base: Direktori dasar input
            output_base: Direktori dasar output
            
        Returns:
            Path output yang sesuai
        """
        # Dapatkan path relatif dari input_base
        rel_path = self.get_relative_path(input_path, input_base)
        
        # Gabungkan dengan output_base
        output_path = output_base / rel_path
        
        # Buat direktori jika belum ada
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def get_temp_dir(self) -> Path:
        """Dapatkan direktori temporary untuk penyimpanan sementara
        
        Returns:
            Path ke direktori temporary
        """
        temp_dir = self.preprocessed_dir / '.temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def clear_temp_dir(self) -> bool:
        """Bersihkan direktori temporary
        
        Returns:
            True jika berhasil, False jika gagal
        """
        temp_dir = self.get_temp_dir()
        try:
            for item in temp_dir.glob('*'):
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
            return True
        except Exception as e:
            return False

"""
File: smartcash/dataset/preprocessor/utils/filename_manager.py
Deskripsi: Modul untuk mengelola penamaan file pada proses preprocessing.
"""
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple


class FilenameManager:
    """Kelas untuk mengelola penamaan file pada proses preprocessing.
    
    Attributes:
        config: Konfigurasi penamaan file
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Inisialisasi FilenameManager.
        
        Args:
            config: Konfigurasi penamaan file (opsional)
        """
        self.config = config or {}
        self._uuid_cache = {}  # Cache untuk menyimpan UUID yang sudah digenerate
    
    def generate_filename(self, original_name: str, split: str = None, 
                          nominal: str = None, increment: int = None) -> str:
        """Generate nama file baru berdasarkan pola yang ditentukan.
        
        Args:
            original_name: Nama file asli
            split: Nama split (train/valid/test)
            nominal: Nilai nominal uang
            increment: Nomor urut
            
        Returns:
            Nama file baru
        """
        # Dapatkan pola dari config atau gunakan default
        pattern = self.config.get('preprocessed_pattern', 'pre_{nominal}_{uuid}_{increment}')
        
        # Generate UUID untuk file ini jika belum ada
        if original_name not in self._uuid_cache:
            self._uuid_cache[original_name] = str(uuid.uuid4().hex[:8])
        
        # Siapkan variabel untuk substitusi
        vars_dict = {
            'split': split or '',
            'nominal': nominal or 'unknown',
            'uuid': self._uuid_cache[original_name],
            'increment': increment or 0,
            'name': Path(original_name).stem
        }
        
        # Lakukan substitusi
        return pattern.format(**vars_dict)
    
    def get_matching_files(self, original_name: str, directory: str) -> Tuple[str, str]:
        """Dapatkan file yang sesuai dengan original_name di direktori.
        
        Args:
            original_name: Nama file asli
            directory: Direktori untuk mencari file
            
        Returns:
            Tuple berisi (nama_file, ekstensi) jika ditemukan, (None, None) jika tidak
        """
        if not os.path.isdir(directory):
            return None, None
            
        # Dapatkan nama file tanpa ekstensi
        stem = Path(original_name).stem
        
        # Cari file dengan stem yang sama
        for f in os.listdir(directory):
            if f.startswith(stem):
                ext = os.path.splitext(f)[1]
                return f, ext
                
        return None, None
    
    def ensure_extension(self, filename: str, ext: str) -> str:
        """Pastikan nama file memiliki ekstensi yang benar.
        
        Args:
            filename: Nama file
            ext: Ekstensi yang diinginkan (dengan atau tanpa titik)
            
        Returns:
            Nama file dengan ekstensi yang benar
        """
        ext = ext if ext.startswith('.') else f'.{ext}'
        base = os.path.splitext(filename)[0]
        return f"{base}{ext}"

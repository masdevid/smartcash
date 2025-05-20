"""
File: smartcash/components/cache/indexing_cache.py
Deskripsi: Modul pengindeksan cache yang dioptimasi dengan DRY principles
"""

import json
import shutil
import threading
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from smartcash.common.logger import get_logger

class CacheIndex:
    """Pengindeksan untuk cache dengan optimasi thread-safety dan efisiensi."""
    
    def __init__(self, cache_dir: Path, logger = None):
        """
        Inisialisasi CacheIndex.
        
        Args:
            cache_dir: Path direktori cache
            logger: Logger kustom (opsional)
        """
        self.cache_dir = Path(cache_dir)
        self.logger = logger or get_logger()
        self.index_path = cache_dir / "cache_index.json"
        self._lock = threading.RLock()
        self.index = {}
        self._init_empty_index()
    
    def _init_empty_index(self) -> None:
        """Inisialisasi struktur index cache kosong."""
        self.index = {
            'files': {},
            'total_size': 0,
            'last_cleanup': None,
            'creation_time': datetime.now().isoformat()
        }
    
    def load_index(self) -> bool:
        """
        Muat index dari file.
        
        Returns:
            Boolean sukses/gagal
        """
        # Gunakan lock untuk thread-safety
        with self._lock:
            # Cek keberadaan file index
            if not self.index_path.exists():
                self._init_empty_index()
                return False
            
            try:
                # Load dan validasi index
                with open(self.index_path, 'r') as f:
                    loaded_index = json.load(f)
                
                # Validasi minimal struktur index
                if not all(key in loaded_index for key in ['files', 'total_size']):
                    self._init_empty_index()
                    return False
                
                # Bersihkan entri tidak valid dengan one-liner
                loaded_index['files'] = {k: v for k, v in loaded_index['files'].items() 
                                       if isinstance(v, dict) and all(key in v for key in ['size', 'timestamp'])}
                
                self.index = loaded_index
                self.logger.info(f"📂 Cache index dimuat: {len(self.index['files'])} entri")
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membaca cache index: {str(e)}")
                self._init_empty_index()
                return False
    
    def save_index(self) -> bool:
        """
        Simpan index ke file dengan aman (atomic write).
        
        Returns:
            Boolean sukses/gagal
        """
        with self._lock:
            # Update timestamp
            self.index['last_updated'] = datetime.now().isoformat()
            temp_path = self.index_path.with_suffix('.tmp')
            
            try:
                # Tulis ke file temporary untuk atomic write
                with open(temp_path, 'w') as f:
                    json.dump(self.index, f, indent=2)
                
                # Pindahkan secara atomic
                shutil.move(str(temp_path), str(self.index_path))
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan cache index: {str(e)}")
                # Hapus file temp jika ada
                [temp_path.unlink() for _ in [1] if temp_path.exists()]
                return False
    
    def get_files(self) -> Dict:
        """
        Dapatkan semua file dalam index.
        
        Returns:
            Dictionary file dalam index
        """
        with self._lock:
            return self.index['files'].copy()
    
    def get_file_info(self, key: str) -> Dict:
        """
        Dapatkan informasi file dengan key tertentu.
        
        Args:
            key: Key file
            
        Returns:
            Dictionary informasi file atau dict kosong jika tidak ada
        """
        with self._lock:
            return self.index['files'].get(key, {}).copy()
    
    def add_file(self, key: str, size: int) -> None:
        """
        Tambahkan file ke index.
        
        Args:
            key: Key file
            size: Ukuran file dalam bytes
        """
        with self._lock:
            # Buat entri baru dengan timestamp
            now = datetime.now().isoformat()
            self.index['files'][key] = {'size': size, 'timestamp': now, 'last_accessed': now}
            
            # Update total ukuran
            self.index['total_size'] += size
            
            # Simpan perubahan
            self.save_index()
    
    def remove_file(self, key: str) -> int:
        """
        Hapus file dari index.
        
        Args:
            key: Key file
            
        Returns:
            Ukuran file yang dihapus (0 jika tidak ada)
        """
        with self._lock:
            # Dapatkan ukuran dan hapus file
            file_info = self.index['files'].pop(key, {})
            size = file_info.get('size', 0)
            
            # Update total ukuran
            if size > 0:
                self.index['total_size'] = max(0, self.index['total_size'] - size)
                self.save_index()
                
            return size
    
    def update_access_time(self, key: str) -> None:
        """
        Update waktu akses terakhir file.
        
        Args:
            key: Key file
        """
        with self._lock:
            # Update last_accessed jika file ada
            if key in self.index['files']:
                self.index['files'][key]['last_accessed'] = datetime.now().isoformat()
                self.save_index()
    
    def update_cleanup_time(self) -> None:
        """Update waktu pembersihan terakhir."""
        with self._lock:
            self.index['last_cleanup'] = datetime.now().isoformat()
            self.save_index()
    
    def get_total_size(self) -> int:
        """
        Dapatkan total ukuran cache dalam index.
        
        Returns:
            Total ukuran dalam bytes
        """
        with self._lock:
            return self.index.get('total_size', 0)
    
    def set_total_size(self, size: int) -> None:
        """
        Set total ukuran cache.
        
        Args:
            size: Ukuran total baru
        """
        with self._lock:
            self.index['total_size'] = size
            self.save_index()
    
    def find_expired_files(self, ttl_seconds: int) -> List[str]:
        """
        Temukan file yang sudah kadaluwarsa.
        
        Args:
            ttl_seconds: Waktu hidup dalam detik
            
        Returns:
            List key file yang kadaluwarsa
        """
        if ttl_seconds <= 0:
            return []
            
        with self._lock:
            now = datetime.now()
            # One-liner untuk mendapatkan file kadaluwarsa
            return [key for key, info in self.index['files'].items() 
                   if (now - datetime.fromisoformat(info['timestamp'])).total_seconds() > ttl_seconds]
    
    def find_least_used_files(self, count: int) -> List[str]:
        """
        Temukan file yang paling jarang diakses.
        
        Args:
            count: Jumlah file yang akan diambil
            
        Returns:
            List key file yang paling jarang diakses
        """
        with self._lock:
            # Urutkan file berdasarkan waktu akses terakhir
            files = [(key, datetime.fromisoformat(info.get('last_accessed', info['timestamp'])))
                    for key, info in self.index['files'].items()]
            
            # Ambil yang paling lama tidak diakses
            files.sort(key=lambda x: x[1])
            return [key for key, _ in files[:count]]
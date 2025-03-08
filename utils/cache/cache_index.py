"""
File: smartcash/utils/cache/cache_index.py
Author: Alfrida Sabar
Deskripsi: Modul untuk mengelola index cache dengan kemampuan persistensi dan validasi.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import threading

from smartcash.utils.logger import SmartCashLogger

class CacheIndex:
    """
    Pengelola index cache dengan dukungan untuk persistensi dan validasi.
    """
    
    def __init__(self, cache_dir: Path, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi pengelola index cache.
        
        Args:
            cache_dir: Direktori cache
            logger: Custom logger
        """
        self.cache_dir = cache_dir
        self.logger = logger or SmartCashLogger(__name__)
        self.index_path = cache_dir / "cache_index.json"
        self._lock = threading.RLock()
        
        # Default index
        self.index = {
            'files': {},
            'total_size': 0,
            'last_cleanup': None,
            'creation_time': datetime.now().isoformat()
        }
    
    def load_index(self) -> bool:
        """
        Load cache index dari disk dengan validasi struktur.
        
        Returns:
            Boolean yang menunjukkan keberhasilan load
        """
        with self._lock:
            if self.index_path.exists():
                try:
                    with open(self.index_path, 'r') as f:
                        self.index = json.load(f)
                    
                    # Validasi struktur index
                    required_keys = ['files', 'total_size', 'last_cleanup']
                    if not all(key in self.index for key in required_keys):
                        raise ValueError("Index struktur tidak valid")
                    
                    # Validasi entri file
                    for key, info in list(self.index['files'].items()):
                        if not all(k in info for k in ['size', 'timestamp']):
                            self.logger.warning(f"⚠️ Entri cache tidak valid: {key}")
                            del self.index['files'][key]
                    
                    self.logger.info(f"📂 Cache index dimuat: {len(self.index['files'])} entri")
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membaca cache index: {str(e)}")
                    self._init_empty_index()
                    return False
            else:
                self._init_empty_index()
                return True
    
    def _init_empty_index(self) -> None:
        """Inisialisasi index cache kosong."""
        self.index = {
            'files': {},
            'total_size': 0,
            'last_cleanup': None,
            'creation_time': datetime.now().isoformat()
        }
        self.logger.info("🆕 Index cache baru dibuat")
    
    def save_index(self) -> bool:
        """
        Simpan cache index ke disk dengan pengaman.
        
        Returns:
            Boolean menunjukkan keberhasilan penyimpanan
        """
        with self._lock:
            # Update timestamp
            self.index['last_updated'] = datetime.now().isoformat()
            
            # Simpan ke file sementara dulu, lalu rename untuk atomic update
            temp_path = self.index_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w') as f:
                    json.dump(self.index, f, indent=2)
                
                # Atomic replace
                shutil.move(str(temp_path), str(self.index_path))
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan cache index: {str(e)}")
                if temp_path.exists():
                    temp_path.unlink()
                return False
    
    def get_files(self) -> Dict:
        """
        Dapatkan dictionary semua file dalam index.
        
        Returns:
            Dictionary berisi semua file cache
        """
        with self._lock:
            return self.index['files']
    
    def get_file_info(self, key: str) -> Dict:
        """
        Dapatkan informasi file dari index.
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary berisi informasi file
        """
        with self._lock:
            return self.index['files'].get(key, {})
    
    def add_file(self, key: str, size: int) -> None:
        """
        Tambahkan file ke index.
        
        Args:
            key: Cache key
            size: Ukuran file dalam bytes
        """
        with self._lock:
            now = datetime.now().isoformat()
            self.index['files'][key] = {
                'size': size,
                'timestamp': now,
                'last_accessed': now
            }
            self.index['total_size'] += size
            self.save_index()
    
    def remove_file(self, key: str) -> int:
        """
        Hapus file dari index.
        
        Args:
            key: Cache key
            
        Returns:
            Ukuran file yang dihapus
        """
        with self._lock:
            if key in self.index['files']:
                size = self.index['files'][key].get('size', 0)
                del self.index['files'][key]
                self.index['total_size'] -= size
                self.save_index()
                return size
            return 0
    
    def update_access_time(self, key: str) -> None:
        """
        Update waktu akses file.
        
        Args:
            key: Cache key
        """
        with self._lock:
            if key in self.index['files']:
                self.index['files'][key]['last_accessed'] = datetime.now().isoformat()
                self.save_index()
    
    def update_cleanup_time(self) -> None:
        """Update waktu cleanup terakhir."""
        with self._lock:
            self.index['last_cleanup'] = datetime.now().isoformat()
            self.save_index()
    
    def get_total_size(self) -> int:
        """
        Dapatkan total ukuran cache.
        
        Returns:
            Total ukuran cache dalam bytes
        """
        with self._lock:
            return self.index.get('total_size', 0)
    
    def set_total_size(self, size: int) -> None:
        """
        Set total ukuran cache.
        
        Args:
            size: Ukuran total dalam bytes
        """
        with self._lock:
            self.index['total_size'] = size
            self.save_index()
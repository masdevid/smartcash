"""
File: smartcash/utils/cache/cache_storage.py
Author: Alfrida Sabar
Deskripsi: Modul untuk operasi penyimpanan dan loading data cache dengan dukungan berbagai format.
"""

import os
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
import threading

from smartcash.utils.logger import SmartCashLogger

class CacheStorage:
    """
    Pengelola penyimpanan dan loading data cache dengan dukungan berbagai format.
    """
    
    def __init__(self, cache_dir: Path, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi pengelola penyimpanan cache.
        
        Args:
            cache_dir: Direktori cache
            logger: Custom logger
        """
        self.cache_dir = cache_dir
        self.logger = logger or SmartCashLogger(__name__)
        self._lock = threading.RLock()
    
    def create_cache_key(self, file_path: str, params: Dict) -> str:
        """
        Generate cache key dari file path dan parameter.
        
        Args:
            file_path: Path file sumber
            params: Parameter preprocessing
            
        Returns:
            String cache key
        """
        # Hash file content untuk ukuran kecil
        # Untuk file besar, hash name+size+mtime untuk performa
        if os.path.getsize(file_path) < 10 * 1024 * 1024:  # < 10 MB
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        else:
            stat = os.stat(file_path)
            content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Hash parameters
        param_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        
        return f"{file_hash[:10]}_{param_hash[:10]}"
    
    def save_to_cache(self, cache_path: Path, data: Any) -> Dict:
        """
        Simpan data ke file cache.
        
        Args:
            cache_path: Path file cache
            data: Data yang akan disimpan
            
        Returns:
            Dict berisi status dan ukuran file
        """
        result = {'success': False, 'size': 0, 'error': None}
        
        try:
            with self._lock:
                # Simpan data ke file
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Dapatkan ukuran file
                result['size'] = cache_path.stat().st_size
                result['success'] = True
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan ke cache: {str(e)}")
            result['error'] = str(e)
            
            # Hapus file jika ada error
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except:
                    pass
                
            return result
    
    def load_from_cache(self, cache_path: Path, measure_time: bool = True) -> Dict:
        """
        Load data dari file cache.
        
        Args:
            cache_path: Path file cache
            measure_time: Jika True, ukur waktu loading
            
        Returns:
            Dict berisi data, status, dan waktu loading
        """
        result = {'success': False, 'data': None, 'load_time': 0, 'error': None}
        
        try:
            # Ukur waktu loading jika diminta
            start_time = time.time() if measure_time else 0
            
            # Load data dari file
            with open(cache_path, 'rb') as f:
                result['data'] = pickle.load(f)
            
            if measure_time:
                result['load_time'] = time.time() - start_time
                
            result['success'] = True
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat membaca cache {cache_path.name}: {str(e)}")
            result['error'] = str(e)
            return result
    
    def delete_file(self, cache_path: Path) -> bool:
        """
        Hapus file cache.
        
        Args:
            cache_path: Path file cache
            
        Returns:
            Boolean menunjukkan keberhasilan penghapusan
        """
        try:
            if cache_path.exists():
                cache_path.unlink()
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Gagal menghapus file cache: {str(e)}")
            return False
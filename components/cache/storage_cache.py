"""
File: smartcash/components/cache/storage_cache.py
Deskripsi: Modul penyimpanan cache yang dioptimasi dengan DRY principles
"""

import os
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count
from smartcash.common.io import ensure_dir, file_exists

class CacheStorage:
    """Utilitas penyimpanan cache dengan optimasi thread."""
    
    def __init__(self, cache_dir: Path, logger = None):
        """
        Inisialisasi CacheStorage.
        
        Args:
            cache_dir: Path direktori cache
            logger: Logger kustom (opsional)
        """
        self.cache_dir = Path(cache_dir)
        self.logger = logger or get_logger("cache_storage")
        self._thread_pool = ThreadPoolExecutor(max_workers=get_optimal_thread_count(), thread_name_prefix="CacheStorage")
    
    def create_cache_key(self, file_path: str, params: Dict) -> str:
        """
        Buat kunci cache berdasarkan file path dan parameter.
        
        Args:
            file_path: Path file sumber
            params: Parameter untuk cache
            
        Returns:
            String kunci cache
        """
        # Hash file berdasarkan ukuran dan timestamp untuk efisiensi
        stat = os.stat(file_path)
        file_content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        file_hash = hashlib.md5(file_content.encode()).hexdigest()[:10]
        
        # Hash parameter dengan pendekatan one-liner
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
        
        return f"{file_hash}_{param_hash}"
    
    def save_to_cache(self, cache_path: Path, data: Any) -> Dict:
        """
        Simpan data ke cache.
        
        Args:
            cache_path: Path file cache
            data: Data yang akan disimpan
            
        Returns:
            Dictionary hasil operasi
        """
        result = {'success': False, 'size': 0, 'error': None}
        
        try:
            # Pastikan direktori cache ada
            ensure_dir(cache_path.parent)
            
            # Pickle dan simpan data secara langsung
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Dapatkan ukuran file hasil
            result.update({'size': cache_path.stat().st_size, 'success': True})
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan ke cache: {str(e)}")
            result['error'] = str(e)
            # Hapus file yang mungkin rusak akibat error
            if cache_path.exists(): [cache_path.unlink() for _ in [1]]
        
        return result
    
    def load_from_cache(self, cache_path: Path, measure_time: bool = True) -> Dict:
        """
        Muat data dari cache.
        
        Args:
            cache_path: Path file cache
            measure_time: Flag untuk mengukur waktu loading
            
        Returns:
            Dictionary hasil operasi
        """
        result = {'success': False, 'data': None, 'load_time': 0, 'error': None}
        
        try:
            # Ukur waktu loading jika diminta
            start_time = time.time() if measure_time else 0
            
            # Load data dari pickle
            with open(cache_path, 'rb') as f:
                result['data'] = pickle.load(f)
            
            # Update waktu loading dan status
            if measure_time: result['load_time'] = time.time() - start_time
            result['success'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Error membaca cache {cache_path.name}: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def save_to_cache_async(self, cache_path: Path, data: Any) -> Any:
        """
        Simpan data ke cache secara asinkron.
        
        Args:
            cache_path: Path file cache
            data: Data yang akan disimpan
            
        Returns:
            Future untuk hasil operasi
        """
        return self._thread_pool.submit(self.save_to_cache, cache_path, data)
    
    def load_from_cache_async(self, cache_path: Path, measure_time: bool = True) -> Any:
        """
        Muat data dari cache secara asinkron.
        
        Args:
            cache_path: Path file cache
            measure_time: Flag untuk mengukur waktu loading
            
        Returns:
            Future untuk hasil operasi
        """
        return self._thread_pool.submit(self.load_from_cache, cache_path, measure_time)
    
    def delete_file(self, cache_path: Path) -> bool:
        """
        Hapus file cache.
        
        Args:
            cache_path: Path file cache
            
        Returns:
            Boolean sukses/gagal
        """
        return file_exists(cache_path) and [cache_path.unlink(), True][1] if cache_path.exists() else False
    
    def shutdown(self):
        """Shutdown thread pool."""
        self._thread_pool.shutdown(wait=True)
"""
File: smartcash/components/cache/cache_storage.py
Deskripsi: Modul untuk file cache_storage.py
"""

import os
import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
import threading

from smartcash.common.logger import SmartCashLogger

class CacheStorage:
    def __init__(self, cache_dir: Path, logger: Optional[SmartCashLogger] = None):
        self.cache_dir = cache_dir
        self.logger = logger or SmartCashLogger(__name__)
        self._lock = threading.RLock()

    def create_cache_key(self, file_path: str, params: Dict) -> str:
        # Hash file berdasarkan ukuran dan timestamp untuk efisiensi
        stat = os.stat(file_path)
        file_content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        file_hash = hashlib.md5(file_content.encode()).hexdigest()[:10]
        
        # Hash parameter
        param_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:10]
        
        return f"{file_hash}_{param_hash}"

    def save_to_cache(self, cache_path: Path, data: Any) -> Dict:
        result = {'success': False, 'size': 0, 'error': None}
        
        try:
            with self._lock:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                
                result['size'] = cache_path.stat().st_size
                result['success'] = True
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan ke cache: {str(e)}")
            result['error'] = str(e)
            
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except: pass
        
        return result

    def load_from_cache(self, cache_path: Path, measure_time: bool = True) -> Dict:
        result = {'success': False, 'data': None, 'load_time': 0, 'error': None}
        
        try:
            start_time = time.time() if measure_time else 0
            
            with open(cache_path, 'rb') as f:
                result['data'] = pickle.load(f)
            
            if measure_time:
                result['load_time'] = time.time() - start_time
                
            result['success'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Error membaca cache {cache_path.name}: {str(e)}")
            result['error'] = str(e)
        
        return result

    def delete_file(self, cache_path: Path) -> bool:
        try:
            if cache_path.exists():
                cache_path.unlink()
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Gagal menghapus file cache: {str(e)}")
            return False
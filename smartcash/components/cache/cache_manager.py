"""
File: smartcash/utils/cache/cache_manager.py
Author: Alfrida Sabar
Deskripsi: Kelas utama mengelola cache dengan TTL, garbage collection, dan monitoring.
"""

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.cache.cache_index import CacheIndex
from smartcash.utils.cache.cache_storage import CacheStorage
from smartcash.utils.cache.cache_cleanup import CacheCleanup
from smartcash.utils.cache.cache_stats import CacheStats

class CacheManager:
    def __init__(
        self,
        cache_dir: str = ".cache/preprocessing",
        max_size_gb: float = 1.0,
        ttl_hours: int = 24,
        auto_cleanup: bool = True,
        cleanup_interval_mins: int = 30,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = ttl_hours * 3600
        
        # Inisialisasi komponen-komponen
        self.cache_index = CacheIndex(self.cache_dir, self.logger)
        self.cache_storage = CacheStorage(self.cache_dir, self.logger)
        self.cache_stats = CacheStats(self.logger)
        
        # Muat index cache
        self.cache_index.load_index()
        
        # Setup cleanup jika diperlukan
        self.cache_cleanup = None
        if auto_cleanup:
            self.cache_cleanup = CacheCleanup(
                self.cache_dir,
                self.cache_index,
                self.max_size_bytes,
                self.ttl_seconds,
                cleanup_interval_mins * 60,
                self.cache_stats,
                self.logger
            )
            self.cache_cleanup.setup_auto_cleanup()

    def get_cache_key(self, file_path: str, params: Dict) -> str:
        return self.cache_storage.create_cache_key(file_path, params)

    def get(self, key: str, measure_time: bool = True) -> Optional[Any]:
        with threading.RLock():
            # Cek keberadaan dan validitas cache
            if key not in self.cache_index.get_files():
                self.cache_stats.update_misses()
                return None
            
            file_info = self.cache_index.get_file_info(key)
            cache_path = self.cache_dir / f"{key}.pkl"
            
            # Validasi file fisik dan masa berlaku
            if not cache_path.exists():
                self.cache_index.remove_file(key)
                self.cache_stats.update_misses()
                return None
            
            # Periksa kadaluwarsa
            if self.ttl_seconds > 0:
                timestamp = datetime.fromisoformat(file_info['timestamp'])
                if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
                    self.cache_stats.update_expired()
                    self.cache_stats.update_misses()
                    return None
            
            # Load data
            result = self.cache_storage.load_from_cache(cache_path, measure_time)
            
            if result['success']:
                self.cache_stats.update_hits()
                if measure_time:
                    self.cache_stats.update_saved_time(result['load_time'])
                
                self.cache_index.update_access_time(key)
                return result['data']
            
            self.cache_stats.update_misses()
            return None

    def put(self, key: str, data: Any, estimated_size: Optional[int] = None) -> bool:
        with threading.RLock():
            # Periksa ukuran cache
            if self.cache_index.get_total_size() > 0.9 * self.max_size_bytes:
                if self.cache_cleanup:
                    self.cache_cleanup.cleanup()
                
                if self.cache_index.get_total_size() > 0.9 * self.max_size_bytes:
                    self.logger.warning("⚠️ Cache penuh, tidak bisa menyimpan")
                    return False
            
            # Simpan data
            cache_path = self.cache_dir / f"{key}.pkl"
            result = self.cache_storage.save_to_cache(cache_path, data)
            
            if result['success']:
                self.cache_index.add_file(key, result['size'])
                self.cache_stats.update_saved_bytes(result['size'])
                return True
            
            return False

    def cleanup(self, expired_only: bool = False, force: bool = False) -> Dict[str, int]:
        return self.cache_cleanup.cleanup(expired_only, force) if self.cache_cleanup else {}

    def clear(self) -> bool:
        return self.cache_cleanup.clear_all() if self.cache_cleanup else False

    def get_stats(self) -> Dict[str, Any]:
        return self.cache_stats.get_all_stats(
            self.cache_dir,
            self.cache_index,
            self.max_size_bytes
        )

    def verify_integrity(self, fix: bool = True) -> Dict[str, int]:
        return self.cache_cleanup.verify_integrity(fix) if self.cache_cleanup else {
            'missing_files': 0, 
            'orphaned_files': 0, 
            'corrupted_files': 0, 
            'fixed': 0
        }
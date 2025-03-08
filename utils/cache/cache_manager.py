"""
File: smartcash/utils/cache/cache_manager.py
Author: Alfrida Sabar
Deskripsi: Kelas utama untuk mengelola cache dengan dukungan untuk TTL, garbage collection, dan monitoring.
"""

from pathlib import Path
from typing import Dict, Optional, Any, Union
from datetime import datetime
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.cache.cache_index import CacheIndex
from smartcash.utils.cache.cache_storage import CacheStorage
from smartcash.utils.cache.cache_cleanup import CacheCleanup
from smartcash.utils.cache.cache_stats import CacheStats

class CacheManager:
    """
    Sistem caching yang ditingkatkan untuk preprocessing dengan:
    - TTL (Time To Live) untuk entri cache
    - Garbage collection otomatis
    - Monitoring penggunaan memori
    - Statistik performa cache
    - Lock untuk operasi multi-threading
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/preprocessing",
        max_size_gb: float = 1.0,
        ttl_hours: int = 24,
        auto_cleanup: bool = True,
        cleanup_interval_mins: int = 30,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi cache manager.
        
        Args:
            cache_dir: Direktori tempat menyimpan cache
            max_size_gb: Ukuran maksimum cache dalam GB
            ttl_hours: Waktu hidup entri cache dalam jam
            auto_cleanup: Aktifkan pembersihan otomatis
            cleanup_interval_mins: Interval pembersihan otomatis dalam menit
            logger: Custom logger
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = ttl_hours * 3600
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval_mins * 60
        
        # Lock untuk thread safety
        self._lock = threading.RLock()
        
        # Inisialisasi komponen-komponen
        self.cache_index = CacheIndex(self.cache_dir, self.logger)
        self.cache_storage = CacheStorage(self.cache_dir, self.logger)
        self.cache_stats = CacheStats(self.logger)
        
        # Muat index cache
        self.cache_index.load_index()
        
        # Setup cleanup jika diperlukan
        if self.auto_cleanup:
            self.cache_cleanup = CacheCleanup(
                self.cache_dir,
                self.cache_index,
                self.max_size_bytes,
                self.ttl_seconds,
                self.cleanup_interval,
                self.cache_stats,
                self.logger
            )
            self.cache_cleanup.setup_auto_cleanup()
    
    def get_cache_key(self, file_path: str, params: Dict) -> str:
        """
        Generate cache key dari file path dan parameter.
        
        Args:
            file_path: Path file sumber
            params: Parameter preprocessing
            
        Returns:
            String cache key
        """
        return self.cache_storage.create_cache_key(file_path, params)
    
    def get(self, key: str, measure_time: bool = True) -> Optional[Any]:
        """
        Ambil hasil preprocessing dari cache.
        
        Args:
            key: Cache key
            measure_time: Jika True, hitung waktu loading
            
        Returns:
            Data hasil preprocessing atau None jika tidak ditemukan
        """
        with self._lock:
            if key not in self.cache_index.get_files():
                self.cache_stats.update_misses()
                return None
            
            # Validasi entri
            file_info = self.cache_index.get_file_info(key)
            cache_path = self.cache_dir / f"{key}.pkl"
            
            # Cek apakah file ada di disk
            if not cache_path.exists():
                self.cache_stats.update_misses()
                self.cache_index.remove_file(key)
                return None
            
            # Cek kadaluwarsa
            if self.ttl_seconds > 0:
                timestamp = datetime.fromisoformat(file_info['timestamp'])
                age = (datetime.now() - timestamp).total_seconds()
                
                if age > self.ttl_seconds:
                    self.cache_stats.update_expired()
                    self.cache_stats.update_misses()
                    # Tidak hapus file di sini, biarkan cleanup yang melakukannya
                    return None
                    
            # Load data
            result = self.cache_storage.load_from_cache(cache_path, measure_time)
            
            if result['success']:
                self.cache_stats.update_hits()
                if measure_time:
                    self.cache_stats.update_saved_time(result['load_time'])
                
                # Update timestamp akses
                self.cache_index.update_access_time(key)
                
                return result['data']
            else:
                self.cache_stats.update_misses()
                return None
    
    def exists(self, key: str) -> bool:
        """
        Cek apakah cache key ada dan valid.
        
        Args:
            key: Cache key
            
        Returns:
            Boolean menunjukkan keberadaan cache
        """
        with self._lock:
            if key not in self.cache_index.get_files():
                return False
                
            file_info = self.cache_index.get_file_info(key)
            cache_path = self.cache_dir / f"{key}.pkl"
            
            if not cache_path.exists():
                return False
                
            # Cek kadaluwarsa
            if self.ttl_seconds > 0:
                timestamp = datetime.fromisoformat(file_info['timestamp'])
                age = (datetime.now() - timestamp).total_seconds()
                
                if age > self.ttl_seconds:
                    return False
                    
            return True
    
    def put(self, key: str, data: Any, estimated_size: Optional[int] = None) -> bool:
        """
        Simpan hasil preprocessing ke cache.
        
        Args:
            key: Cache key
            data: Data yang akan disimpan
            estimated_size: Perkiraan ukuran data (opsional)
            
        Returns:
            Boolean sukses/gagal
        """
        with self._lock:
            # Periksa ukuran cache
            if self.cache_index.get_total_size() > 0.9 * self.max_size_bytes:
                # Cache hampir penuh, bersihkan dulu
                if hasattr(self, 'cache_cleanup'):
                    self.cache_cleanup.cleanup()
                
                # Jika masih penuh, tidak simpan
                if self.cache_index.get_total_size() > 0.9 * self.max_size_bytes:
                    self.logger.warning("⚠️ Cache penuh, tidak bisa menyimpan")
                    return False
            
            # Simpan data ke file
            cache_path = self.cache_dir / f"{key}.pkl"
            result = self.cache_storage.save_to_cache(cache_path, data)
            
            if result['success']:
                # Update index
                self.cache_index.add_file(key, result['size'])
                self.cache_stats.update_saved_bytes(result['size'])
                return True
            else:
                return False
    
    def cleanup(self, expired_only: bool = False, force: bool = False) -> Dict[str, int]:
        """
        Bersihkan cache berdasarkan kadaluarsa atau ukuran.
        
        Args:
            expired_only: Jika True, hanya hapus yang kadaluwarsa
            force: Jika True, bersihkan meskipun di bawah threshold
            
        Returns:
            Dict statistik pembersihan
        """
        if hasattr(self, 'cache_cleanup'):
            return self.cache_cleanup.cleanup(expired_only, force)
        return {'removed': 0, 'freed_bytes': 0, 'errors': 0}
    
    def clear(self) -> bool:
        """
        Bersihkan seluruh cache.
        
        Returns:
            Boolean yang menunjukkan keberhasilan pembersihan
        """
        if hasattr(self, 'cache_cleanup'):
            return self.cache_cleanup.clear_all()
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik cache.
        
        Returns:
            Dict berisi statistik cache
        """
        return self.cache_stats.get_all_stats(
            self.cache_dir,
            self.cache_index,
            self.max_size_bytes
        )
    
    def verify_integrity(self, fix: bool = True) -> Dict[str, int]:
        """
        Verifikasi integritas cache.
        
        Args:
            fix: Jika True, perbaiki masalah yang ditemukan
            
        Returns:
            Dict statistik verifikasi
        """
        if hasattr(self, 'cache_cleanup'):
            return self.cache_cleanup.verify_integrity(fix)
        return {'missing_files': 0, 'orphaned_files': 0, 'corrupted_files': 0, 'fixed': 0}
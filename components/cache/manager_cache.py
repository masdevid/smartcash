"""
File: smartcash/components/cache/manager_cache.py
Deskripsi: Modul manajemen cache yang dioptimasi dengan DRY principles dan ThreadPool
"""

import threading
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.components.cache.indexing_cache import CacheIndex
from smartcash.components.cache.storage_cache import CacheStorage
from smartcash.components.cache.cleanup_cache import CacheCleanup
from smartcash.components.cache.stats_cache import CacheStats
from smartcash.common.io import ensure_dir

class CacheManager:
    """
    Manajer cache terpadu yang mengelola seluruh operasi cache dengan efisiensi tinggi.
    Menggunakan pendekatan modular dengan komponen terpisah untuk setiap fungsi.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/preprocessing",
        max_size_gb: float = 1.0,
        ttl_hours: int = 24,
        auto_cleanup: bool = True,
        cleanup_interval_mins: int = 30,
        logger = None
    ):
        """
        Inisialisasi CacheManager.
        
        Args:
            cache_dir: Path direktori cache
            max_size_gb: Ukuran maksimum cache dalam GB
            ttl_hours: Waktu hidup cache dalam jam
            auto_cleanup: Aktifkan pembersihan otomatis
            cleanup_interval_mins: Interval pembersihan dalam menit
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("cache_manager")
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = ttl_hours * 3600
        
        # Pastikan direktori cache ada
        ensure_dir(self.cache_dir)

        
        # Lock untuk operasi thread-safe
        self._lock = threading.RLock()
        
        # Inisialisasi komponen cache
        self.cache_index = CacheIndex(self.cache_dir, self.logger)
        self.cache_storage = CacheStorage(self.cache_dir, self.logger)
        self.cache_stats = CacheStats(self.logger)
        
        # Muat index cache yang ada
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
        
        # Log inisialisasi
        self.logger.info(
            f"📦 Cache manager diaktifkan: {cache_dir}, "
            f"max={max_size_gb:.1f}GB, TTL={ttl_hours}h"
        )
    
    def get_cache_key(self, file_path: str, params: Dict) -> str:
        """
        Dapatkan kunci cache berdasarkan file path dan parameter.
        
        Args:
            file_path: Path file sumber
            params: Parameter untuk cache
            
        Returns:
            String kunci cache
        """
        return self.cache_storage.create_cache_key(file_path, params)
    
    def get(self, key: str, measure_time: bool = True) -> Optional[Any]:
        """
        Dapatkan data dari cache.
        
        Args:
            key: Kunci cache
            measure_time: Flag untuk mengukur waktu loading
            
        Returns:
            Data atau None jika tidak ada/invalid
        """
        # Thread-safety
        with self._lock:
            # Cek keberadaan cache
            if key not in self.cache_index.get_files():
                self.cache_stats.update_misses()
                return None
            
            # Dapatkan info file dan path
            file_info = self.cache_index.get_file_info(key)
            cache_path = self.cache_dir / f"{key}.pkl"
            
            # Validasi file fisik
            if not cache_path.exists():
                self.cache_index.remove_file(key)
                self.cache_stats.update_misses()
                return None
            
            # Periksa kadaluwarsa jika TTL diaktifkan
            if self.ttl_seconds > 0:
                timestamp = datetime.fromisoformat(file_info['timestamp'])
                if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
                    self.cache_stats.update_expired()
                    self.cache_stats.update_misses()
                    return None
            
            # Load data
            result = self.cache_storage.load_from_cache(cache_path, measure_time)
            
            # Update statistik dan timestamp akses
            if result['success']:
                self.cache_stats.update_hits()
                if measure_time and 'load_time' in result:
                    self.cache_stats.update_saved_time(result['load_time'])
                
                # Update waktu akses terakhir
                self.cache_index.update_access_time(key)
                return result['data']
            
            # Tidak berhasil load
            self.cache_stats.update_misses()
            return None
    
    def put(self, key: str, data: Any, estimated_size: Optional[int] = None) -> bool:
        """
        Simpan data ke cache.
        
        Args:
            key: Kunci cache
            data: Data yang akan disimpan
            estimated_size: Perkiraan ukuran dalam bytes (opsional)
            
        Returns:
            Boolean sukses/gagal
        """
        # Thread-safety
        with self._lock:
            # Periksa ukuran cache dan jalankan cleanup jika hampir penuh
            if self.cache_index.get_total_size() > 0.9 * self.max_size_bytes:
                if self.cache_cleanup:
                    self.cache_cleanup.cleanup()
                
                # Tetap cek lagi setelah cleanup
                if self.cache_index.get_total_size() > 0.9 * self.max_size_bytes:
                    self.logger.warning("⚠️ Cache penuh, tidak bisa menyimpan data baru")
                    return False
            
            # Simpan data
            cache_path = self.cache_dir / f"{key}.pkl"
            result = self.cache_storage.save_to_cache(cache_path, data)
            
            # Update index dan statistik jika berhasil
            if result['success']:
                self.cache_index.add_file(key, result['size'])
                self.cache_stats.update_saved_bytes(result['size'])
                return True
            
            return False
    
    def put_async(self, key: str, data: Any) -> Any:
        """
        Simpan data ke cache secara asinkron.
        
        Args:
            key: Kunci cache
            data: Data yang akan disimpan
            
        Returns:
            Future untuk hasil operasi
        """
        # Simpan dalam thread terpisah
        return self.cache_storage._thread_pool.submit(self.put, key, data)
    
    def remove(self, key: str) -> bool:
        """
        Hapus data dari cache.
        
        Args:
            key: Kunci cache
            
        Returns:
            Boolean sukses/gagal
        """
        with self._lock:
            # Cek keberadaan file
            if key not in self.cache_index.get_files():
                return False
            
            # Hapus file fisik
            cache_path = self.cache_dir / f"{key}.pkl"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    self.logger.error(f"❌ Gagal menghapus file cache {key}: {str(e)}")
                    return False
            
            # Hapus dari index
            size = self.cache_index.remove_file(key)
            
            # Update statistik
            if size > 0:
                self.cache_stats.update_evictions()
                
            return True
    
    def cleanup(self, expired_only: bool = False, force: bool = False) -> Dict[str, int]:
        """
        Bersihkan cache berdasarkan parameter.
        
        Args:
            expired_only: Hanya hapus file kadaluwarsa
            force: Paksa pembersihan walau cache belum terlalu penuh
            
        Returns:
            Dictionary statistik pembersihan
        """
        return self.cache_cleanup.cleanup(expired_only, force) if self.cache_cleanup else {}
    
    def clear(self) -> bool:
        """
        Bersihkan semua cache.
        
        Returns:
            Boolean sukses/gagal
        """
        return self.cache_cleanup.clear_all() if self.cache_cleanup else False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik lengkap cache.
        
        Returns:
            Dictionary statistik
        """
        return self.cache_stats.get_all_stats(
            self.cache_dir,
            self.cache_index,
            self.max_size_bytes
        )
    
    def verify_integrity(self, fix: bool = True) -> Dict[str, int]:
        """
        Verifikasi integritas cache dan perbaiki jika diminta.
        
        Args:
            fix: Perbaiki masalah yang ditemukan
            
        Returns:
            Dictionary statistik verifikasi
        """
        return self.cache_cleanup.verify_integrity(fix) if self.cache_cleanup else {
            'missing_files': 0, 
            'orphaned_files': 0, 
            'corrupted_files': 0, 
            'fixed': 0
        }
    
    def shutdown(self) -> None:
        """Shutdown dan pembersihan sumber daya."""
        with self._lock:
            try:
                # Simpan cache index terakhir
                self.cache_index.save_index()
                
                # Matikan auto-cleanup
                if self.cache_cleanup:
                    self.cache_cleanup.stop_auto_cleanup()
                
                # Shutdown thread pool
                self.cache_storage.shutdown()
                
                self.logger.info("🔌 Cache manager dinonaktifkan")
            except Exception as e:
                self.logger.error(f"❌ Error saat shutdown cache manager: {str(e)}")
    
    def __enter__(self):
        """Context manager untuk penggunaan with-statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup saat keluar context manager."""
        self.shutdown()
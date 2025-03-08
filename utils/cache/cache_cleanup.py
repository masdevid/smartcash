"""
File: smartcash/utils/cache/cache_cleanup.py
Author: Alfrida Sabar
Deskripsi: Modul untuk cleanup dan integriti check pada cache dengan dukungan garbage collection.
"""

import os
import pickle
import time
import threading
import weakref
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any

from smartcash.utils.logger import SmartCashLogger

class CacheCleanup:
    """
    Pengelola cleanup dan integriti check pada cache.
    """
    
    def __init__(
        self, 
        cache_dir: Path,
        cache_index,
        max_size_bytes: int,
        ttl_seconds: int,
        cleanup_interval: int,
        cache_stats,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi pengelola cleanup cache.
        
        Args:
            cache_dir: Direktori cache
            cache_index: Instance CacheIndex
            max_size_bytes: Ukuran maksimum cache dalam bytes
            ttl_seconds: Waktu hidup entri cache dalam detik
            cleanup_interval: Interval pembersihan otomatis dalam detik
            cache_stats: Instance CacheStats
            logger: Custom logger
        """
        self.cache_dir = cache_dir
        self.cache_index = cache_index
        self.max_size_bytes = max_size_bytes
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self.cache_stats = cache_stats
        self.logger = logger or SmartCashLogger(__name__)
        self._lock = threading.RLock()
        self._cleanup_thread = None
    
    def setup_auto_cleanup(self) -> None:
        """Setup thread untuk pembersihan otomatis."""
        def cleanup_worker(cache_ref):
            """Worker function untuk periodic cleanup."""
            cache = cache_ref()
            if cache is None:
                return
                
            time.sleep(60)  # Tunggu 1 menit sebelum mulai
            
            while True:
                try:
                    # Periksa ukuran cache
                    if cache.cache_index.get_total_size() > 0.8 * cache.max_size_bytes:
                        self.logger.info("🧹 Auto cleanup: cache mendekati batas ukuran")
                        cache.cleanup(force=True)
                    else:
                        # Periksa entri yang kadaluwarsa
                        cache.cleanup(expired_only=True)
                        
                    time.sleep(cache.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"❌ Error dalam auto cleanup: {str(e)}")
                    time.sleep(300)  # Tunggu 5 menit jika terjadi error
        
        # Start thread dengan weak reference untuk mencegah memory leak
        cache_ref = weakref.ref(self)
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            args=(cache_ref,),
            daemon=True
        )
        self._cleanup_thread.start()
        self.logger.info(f"🧹 Auto cleanup aktif: interval {self.cleanup_interval/60:.1f} menit")
    
    def cleanup(self, expired_only: bool = False, force: bool = False) -> Dict[str, int]:
        """
        Bersihkan cache berdasarkan kadaluwarsa atau ukuran.
        
        Args:
            expired_only: Jika True, hanya hapus yang kadaluwarsa
            force: Jika True, bersihkan meskipun di bawah threshold
            
        Returns:
            Dict statistik pembersihan
        """
        cleanup_stats = {'removed': 0, 'freed_bytes': 0, 'errors': 0}
        
        with self._lock:
            current_time = datetime.now()
            
            # Identifikasi file yang kadaluwarsa
            expired_keys = self._identify_expired_files(current_time)
            
            # Hapus yang kadaluwarsa
            for key in expired_keys:
                cache_path = self.cache_dir / f"{key}.pkl"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                    
                    size = self.cache_index.get_file_info(key).get('size', 0)
                    self.cache_index.remove_file(key)
                    
                    cleanup_stats['removed'] += 1
                    cleanup_stats['freed_bytes'] += size
                    self.cache_stats.update_evictions()
                except Exception as e:
                    self.logger.error(f"❌ Error saat menghapus {key}: {str(e)}")
                    cleanup_stats['errors'] += 1
            
            # Jika hanya pembersihan expired, selesai di sini
            if expired_only and not force and self.cache_index.get_total_size() <= 0.8 * self.max_size_bytes:
                if cleanup_stats['removed'] > 0:
                    self.logger.info(
                        f"🧹 Removed {cleanup_stats['removed']} expired "
                        f"entries ({cleanup_stats['freed_bytes']/1024/1024:.1f} MB)"
                    )
                self.cache_index.update_cleanup_time()
                return cleanup_stats
            
            # Jika perlu pembersihan lebih lanjut
            if force or self.cache_index.get_total_size() > 0.8 * self.max_size_bytes:
                self._remove_by_lru(cleanup_stats, expired_keys)
            
            # Update index
            self.cache_index.update_cleanup_time()
            
            # Log hasil
            if cleanup_stats['removed'] > 0:
                freed_mb = cleanup_stats['freed_bytes'] / 1024 / 1024
                percent_freed = (cleanup_stats['freed_bytes'] / max(1, self.max_size_bytes)) * 100
                
                self.logger.info(
                    f"🧹 Cache cleanup: {cleanup_stats['removed']} entries removed, "
                    f"{freed_mb:.1f} MB freed ({percent_freed:.1f}%)"
                )
            
            return cleanup_stats
    
    def _identify_expired_files(self, current_time: datetime) -> list:
        """
        Identifikasi file yang kadaluwarsa.
        
        Args:
            current_time: Waktu saat ini
            
        Returns:
            List key file yang kadaluwarsa
        """
        expired_keys = []
        
        if self.ttl_seconds > 0:
            for key, info in self.cache_index.get_files().items():
                timestamp = datetime.fromisoformat(info['timestamp'])
                age = (current_time - timestamp).total_seconds()
                
                if age > self.ttl_seconds:
                    expired_keys.append(key)
                    
        return expired_keys
    
    def clear_all(self) -> bool:
        """
        Bersihkan seluruh cache.
        
        Returns:
            Boolean sukses/gagal
        """
        with self._lock:
            try:
                # Hapus semua file
                for file_path in self.cache_dir.glob("*.pkl"):
                    file_path.unlink()
                
                # Reset index
                self.cache_index._init_empty_index()
                self.cache_index.save_index()
                
                # Reset statistik
                self.cache_stats.reset()
                
                self.logger.success("🧹 Cache berhasil dibersihkan")
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal membersihkan cache: {str(e)}")
                return False
    
    def verify_integrity(self, fix: bool = True) -> Dict[str, int]:
        """
        Verifikasi integritas cache.
        
        Args:
            fix: Jika True, perbaiki masalah yang ditemukan
            
        Returns:
            Dict statistik verifikasi
        """
        with self._lock:
            self.logger.info("🔍 Verifikasi integritas cache...")
            
            stats = {
                'missing_files': 0,
                'orphaned_files': 0,
                'corrupted_files': 0,
                'fixed': 0
            }
            
            # Periksa file yang hilang (dalam index tapi tidak di disk)
            missing_keys = []
            for key in self.cache_index.get_files():
                cache_path = self.cache_dir / f"{key}.pkl"
                if not cache_path.exists():
                    missing_keys.append(key)
                    stats['missing_files'] += 1
            
            # Periksa file yang tidak terdaftar (di disk tapi tidak dalam index)
            index_keys = set(self.cache_index.get_files().keys())
            disk_files = {p.stem for p in self.cache_dir.glob("*.pkl")}
            orphaned_keys = disk_files - index_keys
            stats['orphaned_files'] = len(orphaned_keys)
            
            # Periksa file yang rusak
            corrupted_keys = []
            for key in list(self.cache_index.get_files().keys()):
                if key in missing_keys:
                    continue
                    
                cache_path = self.cache_dir / f"{key}.pkl"
                try:
                    with open(cache_path, 'rb') as f:
                        pickle.load(f)
                except Exception:
                    corrupted_keys.append(key)
                    stats['corrupted_files'] += 1
            
            # Perbaiki masalah jika diminta
            if fix:
                # Hapus entri yang hilang atau rusak dari index
                for key in missing_keys + corrupted_keys:
                    self.cache_index.remove_file(key)
                    stats['fixed'] += 1
                
                # Hapus file yang tidak terdaftar
                for key in orphaned_keys:
                    try:
                        cache_path = self.cache_dir / f"{key}.pkl"
                        cache_path.unlink()
                        stats['fixed'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Gagal menghapus file orphaned {key}: {str(e)}")
                
                self.logger.success(f"✅ Perbaikan cache: {stats['fixed']} masalah diperbaiki")
            
            # Log hasil verifikasi
            self.logger.info(
                f"🔍 Hasil verifikasi cache:\n"
                f"   • Missing files: {stats['missing_files']}\n"
                f"   • Orphaned files: {stats['orphaned_files']}\n"
                f"   • Corrupted files: {stats['corrupted_files']}\n"
                f"   • Total masalah: {sum(stats.values()) - stats['fixed']}"
            )
            
            return stats
    
    def _remove_by_lru(self, cleanup_stats: Dict, already_removed: list) -> None:
        """
        Hapus file berdasarkan metode Least Recently Used.
        
        Args:
            cleanup_stats: Dictionary untuk memperbarui statistik
            already_removed: List key yang sudah dihapus
        """
        # Sort berdasarkan waktu akses
        entries = []
        for key, info in self.cache_index.get_files().items():
            if key in already_removed:
                continue
                
            last_access = info.get('last_accessed', info['timestamp'])
            entries.append((key, datetime.fromisoformat(last_access), info['size']))
        
        # Sort dari akses terlama
        entries.sort(key=lambda x: x[1])
        
        # Hapus sampai ukuran di bawah 70% dari max
        target_size = int(0.7 * self.max_size_bytes)
        
        for key, _, size in entries:
            if self.cache_index.get_total_size() <= target_size:
                break
            
            cache_path = self.cache_dir / f"{key}.pkl"
            try:
                if cache_path.exists():
                    cache_path.unlink()
                
                self.cache_index.remove_file(key)
                
                cleanup_stats['removed'] += 1
                cleanup_stats['freed_bytes'] += size
                self.cache_stats.update_evictions()
            except Exception as e:
                self.logger.error(f"❌ Error saat menghapus {key}: {str(e)}")
                cleanup_stats['errors'] += 1
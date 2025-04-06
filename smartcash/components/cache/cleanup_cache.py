"""
File: smartcash/components/cache/cleanup_cache.py
Deskripsi: Modul pembersihan cache yang dioptimasi dengan DRY principles dan ThreadPool
"""

import time
import pickle
import threading
import weakref
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import process_in_parallel, get_optimal_thread_count

class CacheCleanup:
    """Utilitas pembersihan cache dengan auto-cleanup dan thread-safety."""
    
    def __init__(
        self, 
        cache_dir: Path,
        cache_index,
        max_size_bytes: int,
        ttl_seconds: int,
        cleanup_interval: int,
        cache_stats,
        logger = None
    ):
        """
        Inisialisasi CacheCleanup.
        
        Args:
            cache_dir: Path direktori cache
            cache_index: Instance CacheIndex
            max_size_bytes: Ukuran maksimum cache dalam bytes
            ttl_seconds: Waktu hidup file cache dalam detik
            cleanup_interval: Interval pembersihan otomatis dalam detik
            cache_stats: Instance CacheStats
            logger: Logger kustom (opsional)
        """
        self.cache_dir = cache_dir
        self.cache_index = cache_index
        self.max_size_bytes = max_size_bytes
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self.cache_stats = cache_stats
        self.logger = logger or get_logger("cache_cleanup")
        
        # Setup threading
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max(2, get_optimal_thread_count() // 2), 
            thread_name_prefix="CacheCleanup"
        )
        self._cleanup_thread = None
        self._stop_event = threading.Event()
    
    def setup_auto_cleanup(self) -> None:
        """Mengatur pembersihan otomatis berkala dengan thread daemon."""
        def cleanup_worker(cache_ref):
            cache = cache_ref()
            if not cache: return
            
            # Jeda awal untuk tidak langsung cleanup saat startup
            time.sleep(60)
            
            while not cache._stop_event.is_set():
                try:
                    # Pembersihan: hanya hapus file kadaluwarsa jika cache belum terlalu penuh
                    is_cache_nearly_full = cache.cache_index.get_total_size() > 0.8 * cache.max_size_bytes
                    cache.cleanup(expired_only=not is_cache_nearly_full)
                    
                    # Tunggu hingga interval berikutnya atau sinyal stop
                    cache._stop_event.wait(cache.cleanup_interval)
                except Exception as e:
                    cache.logger.error(f"❌ Auto cleanup error: {str(e)}")
                    # Jeda lebih lama jika error
                    cache._stop_event.wait(300)
        
        # Gunakan weak reference untuk mencegah reference cycle
        self._stop_event.clear()
        cache_ref = weakref.ref(self)
        
        # Buat dan mulai thread daemon
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            args=(cache_ref,),
            daemon=True,
            name="CacheAutoCleanup"
        )
        self._cleanup_thread.start()
        
        self.logger.info(f"🧹 Auto cleanup aktif: interval {self.cleanup_interval/60:.1f} menit")
    
    def stop_auto_cleanup(self) -> None:
        """Hentikan pembersihan otomatis."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_event.set()
            self.logger.info("🛑 Auto cleanup dihentikan")
    
    def cleanup(self, expired_only: bool = False, force: bool = False) -> Dict[str, int]:
        """
        Bersihkan cache berdasarkan parameter.
        
        Args:
            expired_only: Hanya hapus file kadaluwarsa
            force: Paksa pembersihan walau cache belum terlalu penuh
            
        Returns:
            Dictionary statistik pembersihan
        """
        stats = {'removed': 0, 'freed_bytes': 0, 'errors': 0}
        
        with self._lock:
            # Identifikasi file kadaluwarsa
            current_time = datetime.now()
            expired_keys = self._identify_expired_files(current_time)
            
            # Hapus file kadaluwarsa
            for key in expired_keys:
                cache_path = self.cache_dir / f"{key}.pkl"
                try:
                    # Hapus file dan update index
                    if cache_path.exists(): cache_path.unlink()
                    
                    # Dapatkan ukuran file dan update statistik
                    size = self.cache_index.get_file_info(key).get('size', 0)
                    self.cache_index.remove_file(key)
                    
                    # Update statistik
                    stats['removed'] += 1
                    stats['freed_bytes'] += size
                    self.cache_stats.update_evictions()
                except Exception as e:
                    self.logger.error(f"❌ Error menghapus {key}: {str(e)}")
                    stats['errors'] += 1
            
            # Jika hanya expired dan tidak dipaksa, hentikan jika cache belum terlalu penuh
            if expired_only and not force and self.cache_index.get_total_size() <= 0.8 * self.max_size_bytes:
                return stats
            
            # Hapus file berdasarkan LRU jika perlu
            removed_by_lru = self._remove_by_lru(stats)
            stats['removed_by_lru'] = removed_by_lru
            
            # Update waktu cleanup terakhir
            self.cache_index.update_cleanup_time()
            
            # Log statistik pembersihan
            self._log_cleanup_stats(stats)
            
            return stats
    
    def _identify_expired_files(self, current_time: datetime) -> List[str]:
        """
        Identifikasi file yang sudah kadaluwarsa.
        
        Args:
            current_time: Waktu saat ini
            
        Returns:
            List key file yang kadaluwarsa
        """
        # One-liner untuk identifikasi file kadaluwarsa
        return [key for key, info in self.cache_index.get_files().items() 
               if self.ttl_seconds > 0 and 
                  (current_time - datetime.fromisoformat(info['timestamp'])).total_seconds() > self.ttl_seconds]
    
    def _remove_by_lru(self, cleanup_stats: Dict) -> int:
        """
        Hapus file berdasarkan LRU (Least Recently Used).
        
        Args:
            cleanup_stats: Dictionary statistik pembersihan yang akan diupdate
            
        Returns:
            Jumlah file yang dihapus
        """
        # Dapatkan semua entry dan sort berdasarkan waktu akses terakhir
        entries = [(key, datetime.fromisoformat(info.get('last_accessed', info['timestamp'])), info['size'])
                  for key, info in self.cache_index.get_files().items()]
        
        entries.sort(key=lambda x: x[1])  # Sort berdasarkan waktu akses
        
        # Tentukan target ukuran setelah pembersihan (70% dari max)
        target_size = int(0.7 * self.max_size_bytes)
        removed_count = 0
        
        # Hapus file, dimulai dari yang paling jarang diakses sampai mencapai target
        for key, _, size in entries:
            # Hentikan jika sudah mencapai target
            if self.cache_index.get_total_size() <= target_size:
                break
                
            # Hapus file
            cache_path = self.cache_dir / f"{key}.pkl"
            try:
                if cache_path.exists():
                    cache_path.unlink()
                
                # Update index dan statistik
                self.cache_index.remove_file(key)
                cleanup_stats['removed'] += 1
                cleanup_stats['freed_bytes'] += size
                self.cache_stats.update_evictions()
                removed_count += 1
            except Exception as e:
                self.logger.error(f"❌ Error menghapus {key}: {str(e)}")
                cleanup_stats['errors'] += 1
                
        return removed_count
    
    def _log_cleanup_stats(self, stats: Dict) -> None:
        """
        Log statistik pembersihan.
        
        Args:
            stats: Dictionary statistik pembersihan
        """
        if stats['removed'] > 0:
            # Format informasi pembersihan
            freed_mb = stats['freed_bytes'] / (1024 * 1024)
            percent_freed = (stats['freed_bytes'] / max(1, self.max_size_bytes)) * 100
            by_lru = stats.get('removed_by_lru', 0)
            by_expiry = stats['removed'] - by_lru
            
            # Log hasil pembersihan
            self.logger.info(
                f"🧹 Cache cleanup: {stats['removed']} file dihapus "
                f"({by_expiry} expired, {by_lru} LRU), "
                f"{freed_mb:.1f} MB ({percent_freed:.1f}%) dibebaskan"
            )
    
    def clear_all(self) -> bool:
        """
        Bersihkan semua cache.
        
        Returns:
            Boolean sukses/gagal
        """
        with self._lock:
            try:
                # Hapus semua file cache
                removed_count = 0
                total_size = 0
                
                # Gunakan proses paralel untuk penghapusan file
                def remove_file(file_path):
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        return size
                    except Exception:
                        return 0
                
                # Dapatkan semua file cache
                cache_files = list(self.cache_dir.glob("*.pkl"))
                if not cache_files:
                    self.logger.info("✅ Cache sudah kosong")
                    return True
                
                # Hapus file secara paralel
                results = process_in_parallel(
                    cache_files,
                    remove_file,
                    max_workers=self._thread_pool._max_workers,
                    desc="🧹 Menghapus cache",
                    show_progress=len(cache_files) > 100
                )
                
                # Hitung total ukuran yang dihapus
                total_size = sum(results)
                removed_count = len([r for r in results if r > 0])
                
                # Reset index cache
                self.cache_index._init_empty_index()
                self.cache_index.save_index()
                
                # Reset statistik
                self.cache_stats.reset()
                
                # Log hasil
                self.logger.success(
                    f"🧹 Cache berhasil dibersihkan: {removed_count} file, "
                    f"{total_size/(1024*1024):.2f} MB"
                )
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal membersihkan cache: {str(e)}")
                return False
    
    def verify_integrity(self, fix: bool = True) -> Dict[str, int]:
        """
        Verifikasi integritas cache dan perbaiki jika diminta.
        
        Args:
            fix: Perbaiki masalah yang ditemukan
            
        Returns:
            Dictionary statistik verifikasi
        """
        with self._lock:
            self.logger.info("🔍 Verifikasi integritas cache...")
            
            # Statistik verifikasi
            stats = {
                'missing_files': 0,
                'orphaned_files': 0,
                'corrupted_files': 0,
                'fixed': 0
            }
            
            # Proses verifikasi
            index_keys = set(self.cache_index.get_files().keys())
            disk_files = {p.stem for p in self.cache_dir.glob("*.pkl")}
            
            # Cek file yang hilang/rusak dalam index
            for key in list(index_keys):
                cache_path = self.cache_dir / f"{key}.pkl"
                
                if not cache_path.exists():
                    stats['missing_files'] += 1
                    if fix:
                        self.cache_index.remove_file(key)
                        stats['fixed'] += 1
                elif not self._is_valid_pickle(cache_path):
                    stats['corrupted_files'] += 1
                    if fix:
                        cache_path.unlink()
                        self.cache_index.remove_file(key)
                        stats['fixed'] += 1
            
            # Cek file yang tidak terdaftar di index
            orphaned_keys = disk_files - index_keys
            stats['orphaned_files'] = len(orphaned_keys)
            
            # Perbaiki file orphaned
            if fix and orphaned_keys:
                for key in orphaned_keys:
                    orphan_path = self.cache_dir / f"{key}.pkl"
                    
                    # Hapus file orphan
                    if orphan_path.exists():
                        try:
                            orphan_path.unlink()
                            stats['fixed'] += 1
                        except Exception as e:
                            self.logger.error(f"❌ Gagal menghapus file orphan {key}: {str(e)}")
            
            # Rekalkulasi total ukuran jika diperbaiki
            if fix and (stats['missing_files'] > 0 or stats['corrupted_files'] > 0):
                actual_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
                self.cache_index.set_total_size(actual_size)
            
            # Log hasil
            total_issues = stats['missing_files'] + stats['orphaned_files'] + stats['corrupted_files']
            if total_issues > 0:
                msg = f"🔍 Menemukan {total_issues} masalah pada cache"
                if fix:
                    msg += f", {stats['fixed']} diperbaiki"
                self.logger.warning(msg)
            else:
                self.logger.info("✅ Integritas cache baik, tidak ada masalah")
            
            return stats
    
    def _is_valid_pickle(self, path: Path) -> bool:
        """
        Cek apakah file pickle valid.
        
        Args:
            path: Path file pickle
            
        Returns:
            Boolean yang menunjukkan validitas
        """
        try:
            with open(path, 'rb') as f:
                pickle.load(f)
            return True
        except Exception:
            return False
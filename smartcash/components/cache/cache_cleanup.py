"""
File: smartcash/components/cache/cache_cleanup.py
Deskripsi: Modul untuk file cache_cleanup.py
"""

import time
import threading
import weakref
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from smartcash.common.logger import SmartCashLogger

class CacheCleanup:
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
        self.cache_dir, self.cache_index = cache_dir, cache_index
        self.max_size_bytes = max_size_bytes
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self.cache_stats = cache_stats
        self.logger = logger or SmartCashLogger(__name__)
        self._lock = threading.RLock()

    def setup_auto_cleanup(self):
        def cleanup_worker(cache_ref):
            cache = cache_ref()
            if not cache: return
            
            time.sleep(60)
            while True:
                try:
                    cache.cleanup(
                        expired_only=cache.cache_index.get_total_size() <= 0.8 * cache.max_size_bytes
                    )
                    time.sleep(cache.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"❌ Auto cleanup error: {str(e)}")
                    time.sleep(300)
        
        cache_ref = weakref.ref(self)
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            args=(cache_ref,),
            daemon=True
        )
        self._cleanup_thread.start()
        self.logger.info(f"🧹 Auto cleanup aktif: {self.cleanup_interval/60:.1f} menit")

    def cleanup(self, expired_only: bool = False, force: bool = False) -> Dict[str, int]:
        stats = {'removed': 0, 'freed_bytes': 0, 'errors': 0}
        
        with self._lock:
            current_time = datetime.now()
            expired_keys = self._identify_expired_files(current_time)
            
            for key in expired_keys:
                cache_path = self.cache_dir / f"{key}.pkl"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                    
                    size = self.cache_index.get_file_info(key).get('size', 0)
                    self.cache_index.remove_file(key)
                    
                    stats['removed'] += 1
                    stats['freed_bytes'] += size
                    self.cache_stats.update_evictions()
                except Exception as e:
                    self.logger.error(f"❌ Error menghapus {key}: {str(e)}")
                    stats['errors'] += 1
            
            if expired_only and not force and self.cache_index.get_total_size() <= 0.8 * self.max_size_bytes:
                return stats
            
            self._remove_by_lru(stats, expired_keys)
            self.cache_index.update_cleanup_time()
            
            self._log_cleanup_stats(stats)
            return stats

    def _identify_expired_files(self, current_time: datetime) -> list:
        return [
            key for key, info in self.cache_index.get_files().items()
            if self.ttl_seconds > 0 and 
            (current_time - datetime.fromisoformat(info['timestamp'])).total_seconds() > self.ttl_seconds
        ]

    def _remove_by_lru(self, cleanup_stats: Dict, already_removed: list):
        entries = [
            (key, datetime.fromisoformat(info.get('last_accessed', info['timestamp'])), info['size'])
            for key, info in self.cache_index.get_files().items()
            if key not in already_removed
        ]
        
        entries.sort(key=lambda x: x[1])
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
                self.logger.error(f"❌ Error menghapus {key}: {str(e)}")
                cleanup_stats['errors'] += 1

    def _log_cleanup_stats(self, stats: Dict):
        if stats['removed'] > 0:
            freed_mb = stats['freed_bytes'] / 1024 / 1024
            percent_freed = (stats['freed_bytes'] / max(1, self.max_size_bytes)) * 100
            
            self.logger.info(
                f"🧹 Cache cleanup: {stats['removed']} entries removed, "
                f"{freed_mb:.1f} MB freed ({percent_freed:.1f}%)"
            )

    def clear_all(self) -> bool:
        with self._lock:
            try:
                for file_path in self.cache_dir.glob("*.pkl"):
                    file_path.unlink()
                
                self.cache_index._init_empty_index()
                self.cache_index.save_index()
                self.cache_stats.reset()
                
                self.logger.success("🧹 Cache berhasil dibersihkan")
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal membersihkan cache: {str(e)}")
                return False

    def verify_integrity(self, fix: bool = True) -> Dict[str, int]:
        with self._lock:
            self.logger.info("🔍 Verifikasi integritas cache...")
            
            stats = {
                'missing_files': 0,
                'orphaned_files': 0,
                'corrupted_files': 0,
                'fixed': 0
            }
            
            # Proses verifikasi dan perbaikan
            index_keys = set(self.cache_index.get_files().keys())
            disk_files = {p.stem for p in self.cache_dir.glob("*.pkl")}
            
            # Cek file yang hilang/rusak
            for key in list(self.cache_index.get_files().keys()):
                cache_path = self.cache_dir / f"{key}.pkl"
                try:
                    with open(cache_path, 'rb') as f:
                        pickle.load(f)
                except Exception:
                    if cache_path.exists():
                        stats['corrupted_files'] += 1
                    else:
                        stats['missing_files'] += 1
            
            # Cek file yang tidak terdaftar
            stats['orphaned_files'] = len(disk_files - index_keys)
            
            # Perbaiki masalah jika diminta
            if fix:
                for key in list(index_keys):
                    cache_path = self.cache_dir / f"{key}.pkl"
                    if not cache_path.exists() or not self._is_valid_pickle(cache_path):
                        self.cache_index.remove_file(key)
                        stats['fixed'] += 1
                
                for key in disk_files - index_keys:
                    (self.cache_dir / f"{key}.pkl").unlink()
                    stats['fixed'] += 1
                
                self.logger.success(f"✅ Perbaikan cache: {stats['fixed']} masalah diperbaiki")
            
            self._log_integrity_stats(stats)
            return stats

    def _is_valid_pickle(self, path: Path) -> bool:
        try:
            with open(path, 'rb') as f:
                pickle.load(f)
            return True
        except:
            return False

    def _log_integrity_stats(self, stats: Dict):
        self.logger.info(
            f"🔍 Hasil verifikasi cache:\n"
            f"   • Missing files: {stats['missing_files']}\n"
            f"   • Orphaned files: {stats['orphaned_files']}\n"
            f"   • Corrupted files: {stats['corrupted_files']}\n"
            f"   • Total masalah: {sum(stats.values()) - stats['fixed']}"
        )
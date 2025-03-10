"""
File: smartcash/utils/cache/cache_stats.py
Author: Alfrida Sabar
Deskripsi: Modul untuk pengelolaan statistik cache dengan dukungan metrik performa.
"""

import threading
from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger

class CacheStats:
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        self.logger = logger or SmartCashLogger(__name__)
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._stats = {
                'hits': 0, 'misses': 0, 'evictions': 0,
                'expired': 0, 'saved_bytes': 0, 'saved_time': 0.0
            }

    def update_hits(self) -> None:
        with self._lock: self._stats['hits'] += 1

    def update_misses(self) -> None:
        with self._lock: self._stats['misses'] += 1

    def update_evictions(self) -> None:
        with self._lock: self._stats['evictions'] += 1

    def update_expired(self) -> None:
        with self._lock: self._stats['expired'] += 1

    def update_saved_bytes(self, bytes_saved: int) -> None:
        with self._lock: self._stats['saved_bytes'] += bytes_saved

    def update_saved_time(self, time_saved: float) -> None:
        with self._lock: self._stats['saved_time'] += time_saved

    def get_raw_stats(self) -> Dict[str, Any]:
        with self._lock: return dict(self._stats)

    def get_all_stats(self, cache_dir: Path, cache_index, max_size_bytes: int) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_ratio = (self._stats['hits'] / max(1, total_requests)) * 100
            
            file_count = len(list(cache_dir.glob("*.pkl")))
            
            # Validasi ukuran aktual
            actual_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
            index_size = cache_index.get_total_size()
            
            if abs(actual_size - index_size) > 1024 * 1024:
                self.logger.warning(
                    f"⚠️ Perbedaan ukuran cache: "
                    f"index={index_size/1024/1024:.1f}MB, "
                    f"disk={actual_size/1024/1024:.1f}MB"
                )
                cache_index.set_total_size(actual_size)
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_ratio': hit_ratio,
                'evictions': self._stats['evictions'],
                'expired': self._stats['expired'],
                'saved_bytes': self._stats['saved_bytes'],
                'saved_time': self._stats['saved_time'],
                'cache_size_bytes': index_size,
                'cache_size_mb': index_size / 1024 / 1024,
                'max_size_mb': max_size_bytes / 1024 / 1024,
                'usage_percent': (index_size / max(1, max_size_bytes)) * 100,
                'file_count': file_count
            }
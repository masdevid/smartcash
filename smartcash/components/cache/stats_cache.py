"""
File: smartcash/components/cache/stats_cache.py
Deskripsi: Modul statistik cache yang dioptimasi dengan DRY principles
"""

import threading
from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.common.io import format_size
from smartcash.common.logger import get_logger

class CacheStats:
    """Utilitas statistik cache dengan thread-safety dan laporan statistik lengkap."""
    
    def __init__(self, logger = None):
        """
        Inisialisasi CacheStats.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger()
        self._lock = threading.RLock()
        self.reset()
    
    def reset(self) -> None:
        """Reset semua statistik ke nilai awal."""
        with self._lock:
            self._stats = {
                'hits': 0, 'misses': 0, 'evictions': 0, 'expired': 0, 
                'saved_bytes': 0, 'saved_time': 0.0, 'errors': 0,
                'last_updated': None
            }
    
    def _update_stat(self, key: str, value: Any = 1, increment: bool = True) -> None:
        """
        Update statistik dengan thread-safety.
        
        Args:
            key: Kunci statistik
            value: Nilai yang akan ditambahkan/diset
            increment: True untuk menambah, False untuk mengeset
        """
        with self._lock:
            if increment and key in self._stats and isinstance(self._stats[key], (int, float)):
                self._stats[key] += value
            else:
                self._stats[key] = value
    
    # Semua method update menggunakan _update_stat untuk mengurangi redundansi
    def update_hits(self) -> None:
        """Tambah statistik hit cache."""
        self._update_stat('hits')
    
    def update_misses(self) -> None:
        """Tambah statistik miss cache."""
        self._update_stat('misses')
    
    def update_evictions(self) -> None:
        """Tambah statistik eviction cache."""
        self._update_stat('evictions')
    
    def update_expired(self) -> None:
        """Tambah statistik expired cache."""
        self._update_stat('expired')
    
    def update_errors(self) -> None:
        """Tambah statistik error cache."""
        self._update_stat('errors')
    
    def update_saved_bytes(self, bytes_saved: int) -> None:
        """
        Tambah statistik byte yang disimpan.
        
        Args:
            bytes_saved: Jumlah byte yang disimpan
        """
        self._update_stat('saved_bytes', bytes_saved)
    
    def update_saved_time(self, time_saved: float) -> None:
        """
        Tambah statistik waktu yang dihemat.
        
        Args:
            time_saved: Waktu dalam detik yang dihemat
        """
        self._update_stat('saved_time', time_saved)
    
    def get_raw_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik mentah.
        
        Returns:
            Dictionary statistik
        """
        with self._lock:
            return dict(self._stats)
    
    def get_all_stats(self, cache_dir: Path, cache_index, max_size_bytes: int) -> Dict[str, Any]:
        """
        Dapatkan statistik lengkap dengan informasi file dan disk.
        
        Args:
            cache_dir: Path direktori cache
            cache_index: Instance CacheIndex
            max_size_bytes: Ukuran maksimum cache dalam bytes
            
        Returns:
            Dictionary statistik lengkap
        """
        with self._lock:
            # Perhitungan dasar
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_ratio = (self._stats['hits'] / max(1, total_requests)) * 100
            
            # Hitung file secara langsung dari filesystem
            file_count = len(list(cache_dir.glob("*.pkl")))
            index_size = cache_index.get_total_size()
            
            # Validasi ukuran cache dengan real filesystem
            actual_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
            
            # Jika ada perbedaan signifikan, log warning
            if abs(actual_size - index_size) > 1024 * 1024:  # > 1MB
                self.logger.warning(
                    f"⚠️ Perbedaan ukuran cache: "
                    f"index={format_size(index_size)}, "
                    f"disk={format_size(actual_size)}"
                )
                cache_index.set_total_size(actual_size)
            
            # Satu-satunya one-liner: satu return menggabungkan semua metrik
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_ratio': hit_ratio,
                'evictions': self._stats['evictions'],
                'expired': self._stats['expired'],
                'errors': self._stats.get('errors', 0),
                'saved_bytes': self._stats['saved_bytes'],
                'saved_time': self._stats['saved_time'],
                'cache_size_bytes': index_size,
                'cache_size_mb': index_size / 1024 / 1024,
                'cache_size_formatted': format_size(index_size),
                'max_size_mb': max_size_bytes / 1024 / 1024,
                'max_size_formatted': format_size(max_size_bytes),
                'usage_percent': (index_size / max(1, max_size_bytes)) * 100,
                'file_count': file_count,
                'efficiency': {
                    'bytes_per_hit': self._stats['saved_bytes'] / max(1, self._stats['hits']),
                    'time_per_hit': self._stats['saved_time'] / max(1, self._stats['hits']),
                    'eviction_ratio': self._stats['evictions'] / max(1, total_requests) * 100,
                    'error_ratio': self._stats.get('errors', 0) / max(1, total_requests) * 100
                }
            }
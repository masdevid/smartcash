"""
File: smartcash/dataset/services/loader/cache_manager.py
Deskripsi: Manager cache untuk dataset yang mengoptimalkan penyimpanan dan akses data
"""

import os
import gc
import pickle
import hashlib
import threading
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime

import torch
from smartcash.common.logger import get_logger


class DatasetCacheManager:
    """
    Manager cache untuk dataset dengan dukungan untuk penyimpanan RAM dan disk.
    Mengoptimalkan penggunaan memori dan kecepatan akses data.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_ram_usage_gb: float = 2.0,
        max_disk_usage_gb: float = 10.0,
        ttl_seconds: int = 3600 * 24,  # 1 hari
        logger=None
    ):
        """
        Inisialisasi DatasetCacheManager.
        
        Args:
            cache_dir: Direktori untuk cache disk (opsional)
            max_ram_usage_gb: Batas penggunaan RAM dalam GB
            max_disk_usage_gb: Batas penggunaan disk dalam GB
            ttl_seconds: Waktu hidup cache dalam detik
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("dataset_cache")
        self.max_ram_usage_bytes = int(max_ram_usage_gb * 1024 * 1024 * 1024)
        self.max_disk_usage_bytes = int(max_disk_usage_gb * 1024 * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.expanduser("~")) / ".smartcash" / "cache"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache RAM dan metadata
        self._ram_cache = {}  # key -> data
        self._metadata = {}   # key -> {'size': bytes, 'last_access': timestamp, 'hits': count}
        self._current_ram_usage = 0
        
        # Lock untuk thread safety
        self._lock = threading.RLock()
        
        # Flag untuk periodic cleanup
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 menit
        
        self.logger.info(
            f"💾 DatasetCacheManager diinisialisasi:\n"
            f"   • Direktori: {self.cache_dir}\n"
            f"   • Batas RAM: {max_ram_usage_gb:.1f} GB\n"
            f"   • Batas Disk: {max_disk_usage_gb:.1f} GB\n"
            f"   • TTL: {ttl_seconds/3600:.1f} jam"
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Ambil item dari cache (RAM atau disk).
        
        Args:
            key: Kunci cache
            default: Nilai default jika item tidak ditemukan
            
        Returns:
            Data dari cache atau default
        """
        hash_key = self._hash_key(key)
        
        with self._lock:
            # Check periodic cleanup
            self._check_periodic_cleanup()
            
            # Cek RAM cache dulu
            if hash_key in self._ram_cache:
                self._update_metadata(hash_key, 'hit')
                return self._ram_cache[hash_key]
            
            # Coba ambil dari disk
            disk_path = self.cache_dir / f"{hash_key}.pkl"
            if disk_path.exists():
                try:
                    data = self._load_from_disk(disk_path)
                    # Tambahkan ke RAM cache jika ada ruang
                    self._try_cache_to_ram(hash_key, data)
                    return data
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal load cache dari disk: {str(e)}")
                    
        return default
    
    def put(self, key: str, data: Any, force: bool = False) -> bool:
        """
        Simpan item ke cache.
        
        Args:
            key: Kunci cache
            data: Data yang akan disimpan
            force: Apakah memaksa penyimpanan meskipun melebihi batas
            
        Returns:
            Sukses atau tidak
        """
        hash_key = self._hash_key(key)
        
        # Hitung ukuran data
        try:
            data_size = self._estimate_size(data)
        except Exception:
            data_size = 0
            
        # Skip jika terlalu besar dan tidak force
        if not force and data_size > self.max_ram_usage_bytes * 0.5:
            self.logger.warning(f"⚠️ Data terlalu besar untuk cache: {data_size / (1024*1024):.1f} MB")
            return False
        
        with self._lock:
            # Simpan ke RAM jika ada ruang
            if self._try_cache_to_ram(hash_key, data):
                # Juga simpan ke disk untuk persistensi
                if data_size < self.max_disk_usage_bytes * 0.1:  # Pastikan tidak terlalu besar
                    self._save_to_disk(hash_key, data)
                return True
            
            # Jika tidak ada ruang di RAM, hanya simpan ke disk
            if data_size < self.max_disk_usage_bytes * 0.1:
                return self._save_to_disk(hash_key, data)
                
            return False
    
    def clear(self, older_than_hours: Optional[float] = None) -> int:
        """
        Bersihkan cache.
        
        Args:
            older_than_hours: Hanya hapus cache yang lebih tua dari jam ini (opsional)
            
        Returns:
            Jumlah item yang dihapus
        """
        with self._lock:
            # Hitung timestamp untuk filter
            if older_than_hours is not None:
                cutoff_time = time.time() - (older_than_hours * 3600)
            else:
                cutoff_time = float('inf')  # Hapus semua
            
            # Hapus dari RAM
            keys_to_remove = []
            for key, meta in self._metadata.items():
                if meta.get('last_access', 0) < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self._ram_cache:
                    data_size = self._metadata[key].get('size', 0)
                    del self._ram_cache[key]
                    del self._metadata[key]
                    self._current_ram_usage -= data_size
            
            ram_removed = len(keys_to_remove)
            
            # Hapus dari disk
            disk_removed = 0
            for file_path in self.cache_dir.glob("*.pkl"):
                try:
                    # Cek waktu modifikasi file
                    mtime = file_path.stat().st_mtime
                    if mtime < cutoff_time:
                        file_path.unlink()
                        disk_removed += 1
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal hapus cache file: {str(e)}")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info(
                f"🧹 Cache dibersihkan:\n"
                f"   • RAM: {ram_removed} item\n"
                f"   • Disk: {disk_removed} file"
            )
            
            return ram_removed + disk_removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik cache.
        
        Returns:
            Dictionary berisi statistik cache
        """
        with self._lock:
            # Hitung statistik RAM
            ram_items = len(self._ram_cache)
            ram_usage = self._current_ram_usage
            ram_usage_percent = (ram_usage / max(1, self.max_ram_usage_bytes)) * 100
            
            # Hitung statistik disk
            disk_items = len(list(self.cache_dir.glob("*.pkl")))
            disk_usage = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            disk_usage_percent = (disk_usage / max(1, self.max_disk_usage_bytes)) * 100
            
            # Hitung hit stats
            total_hits = sum(meta.get('hits', 0) for meta in self._metadata.values())
            total_accesses = sum(1 for meta in self._metadata.values())
            hit_ratio = total_hits / max(1, total_accesses)
            
            return {
                'ram_items': ram_items,
                'ram_usage_bytes': ram_usage,
                'ram_usage_mb': ram_usage / (1024 * 1024),
                'ram_usage_percent': ram_usage_percent,
                'disk_items': disk_items,
                'disk_usage_bytes': disk_usage,
                'disk_usage_mb': disk_usage / (1024 * 1024),
                'disk_usage_percent': disk_usage_percent,
                'hit_ratio': hit_ratio,
                'total_hits': total_hits
            }
    
    def _hash_key(self, key: str) -> str:
        """Generate hash untuk kunci cache."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _estimate_size(self, data: Any) -> int:
        """Estimasi ukuran data dalam bytes."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            # Estimasi untuk array/tensor
            return data.nbytes if hasattr(data, 'nbytes') else 0
        elif isinstance(data, (list, tuple, dict, set)):
            # Rekursif untuk container
            if len(data) > 1000:  # Hanya sampling untuk container besar
                if isinstance(data, dict):
                    sample_keys = list(data.keys())[:10]
                    sample_size = sum(self._estimate_size(data[k]) for k in sample_keys)
                    return int(sample_size * (len(data) / len(sample_keys)))
                else:
                    sample = data[:10]
                    sample_size = sum(self._estimate_size(item) for item in sample)
                    return int(sample_size * (len(data) / len(sample)))
            else:
                if isinstance(data, dict):
                    return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
                else:
                    return sum(self._estimate_size(item) for item in data)
        else:
            # Fallback untuk tipe lain
            return sys.getsizeof(data)
    
    def _update_metadata(self, key: str, action: str = 'access') -> None:
        """Update metadata untuk kunci cache."""
        if key not in self._metadata:
            self._metadata[key] = {'size': 0, 'hits': 0, 'last_access': time.time()}
            
        self._metadata[key]['last_access'] = time.time()
        
        if action == 'hit':
            self._metadata[key]['hits'] = self._metadata[key].get('hits', 0) + 1
    
    def _try_cache_to_ram(self, key: str, data: Any) -> bool:
        """
        Coba simpan data ke RAM cache jika ada ruang.
        
        Args:
            key: Kunci cache
            data: Data yang akan disimpan
            
        Returns:
            Sukses atau tidak
        """
        try:
            # Estimasi ukuran
            data_size = self._estimate_size(data)
            
            # Cek kapasitas
            if key in self._ram_cache:
                # Update existing
                old_size = self._metadata[key].get('size', 0)
                self._current_ram_usage = max(0, self._current_ram_usage - old_size)
                self._ram_cache[key] = data
                self._metadata[key]['size'] = data_size
                self._current_ram_usage += data_size
                self._update_metadata(key)
                return True
            elif self._current_ram_usage + data_size <= self.max_ram_usage_bytes:
                # Add new
                self._ram_cache[key] = data
                self._metadata[key] = {'size': data_size, 'hits': 0, 'last_access': time.time()}
                self._current_ram_usage += data_size
                return True
            else:
                # Tidak cukup ruang, coba evict item lama
                freed_space = self._evict_ram_cache(data_size)
                if self._current_ram_usage + data_size - freed_space <= self.max_ram_usage_bytes:
                    self._ram_cache[key] = data
                    self._metadata[key] = {'size': data_size, 'hits': 0, 'last_access': time.time()}
                    self._current_ram_usage += data_size
                    return True
                    
            return False
        except Exception as e:
            self.logger.debug(f"⚠️ Error saat cache ke RAM: {str(e)}")
            return False
    
    def _evict_ram_cache(self, needed_space: int) -> int:
        """
        Evict item dari RAM cache berdasarkan LRU untuk membuat ruang.
        
        Args:
            needed_space: Ruang yang dibutuhkan dalam bytes
            
        Returns:
            Ruang yang berhasil dibebaskan
        """
        # Sort berdasarkan last_access (oldest first)
        sorted_keys = sorted(
            self._metadata.keys(),
            key=lambda k: self._metadata[k].get('last_access', 0)
        )
        
        freed_space = 0
        for key in sorted_keys:
            if key in self._ram_cache:
                freed_space += self._metadata[key].get('size', 0)
                del self._ram_cache[key]
                # Jangan hapus metadata, karena mungkin masih ada di disk
                
                if freed_space >= needed_space:
                    break
                    
        return freed_space
    
    def _save_to_disk(self, key: str, data: Any) -> bool:
        """
        Simpan data ke disk cache.
        
        Args:
            key: Kunci cache
            data: Data yang akan disimpan
            
        Returns:
            Sukses atau tidak
        """
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=4)  # Protocol 4 untuk kompatibilitas
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ Gagal menyimpan cache ke disk: {str(e)}")
            return False
    
    def _load_from_disk(self, file_path: Path) -> Any:
        """
        Load data dari disk cache.
        
        Args:
            file_path: Path ke file cache
            
        Returns:
            Data dari cache
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _check_periodic_cleanup(self) -> None:
        """Cek dan jalankan periodic cleanup jika diperlukan."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._last_cleanup = current_time
            
            # Hitung expired items (TTL exceeded)
            expired_time = current_time - self.ttl_seconds
            
            # Cleanup RAM
            expired_keys = [
                key for key, meta in self._metadata.items()
                if meta.get('last_access', 0) < expired_time
            ]
            
            for key in expired_keys:
                if key in self._ram_cache:
                    self._current_ram_usage -= self._metadata[key].get('size', 0)
                    del self._ram_cache[key]
                    del self._metadata[key]
            
            # Cleanup disk
            for file_path in self.cache_dir.glob("*.pkl"):
                try:
                    if file_path.stat().st_mtime < expired_time:
                        file_path.unlink()
                except Exception:
                    pass
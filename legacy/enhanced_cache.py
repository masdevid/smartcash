# File: smartcash/utils/enhanced_cache.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi cache yang lebih efisien dengan TTL, garbage collection, dan statistik performa

import os
import pickle
import hashlib
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple, List
from datetime import datetime, timedelta
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger

class EnhancedCache:
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
        Inisialisasi cache yang ditingkatkan.
        
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
        
        # Statistik cache
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0,
            'saved_bytes': 0,
            'saved_time': 0.0
        }
        
        # Load index cache
        self.index_path = self.cache_dir / "cache_index.json"
        self._load_index()
        
        # Jalankan thread garbage collection jika auto cleanup aktif
        if self.auto_cleanup:
            self._setup_auto_cleanup()
    
    def _load_index(self) -> None:
        """Load cache index dari disk dengan validasi struktur."""
        with self._lock:
            if self.index_path.exists():
                try:
                    with open(self.index_path, 'r') as f:
                        self.index = json.load(f)
                    
                    # Validasi struktur index
                    required_keys = ['files', 'total_size', 'last_cleanup']
                    if not all(key in self.index for key in required_keys):
                        raise ValueError("Index struktur tidak valid")
                    
                    # Validasi entri file
                    for key, info in list(self.index['files'].items()):
                        if not all(k in info for k in ['size', 'timestamp']):
                            self.logger.warning(f"⚠️ Entri cache tidak valid: {key}")
                            del self.index['files'][key]
                    
                    self.logger.info(f"📂 Cache index dimuat: {len(self.index['files'])} entri")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal membaca cache index: {str(e)}")
                    self._init_empty_index()
            else:
                self._init_empty_index()
    
    def _init_empty_index(self) -> None:
        """Inisialisasi index cache kosong."""
        self.index = {
            'files': {},
            'total_size': 0,
            'last_cleanup': None,
            'creation_time': datetime.now().isoformat()
        }
        self.logger.info("🆕 Index cache baru dibuat")
    
    def _setup_auto_cleanup(self) -> None:
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
                    if cache.index['total_size'] > 0.8 * cache.max_size_bytes:
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
    
    def save_index(self) -> None:
        """Simpan cache index ke disk dengan pengaman."""
        with self._lock:
            # Update timestamp
            self.index['last_updated'] = datetime.now().isoformat()
            
            # Simpan ke file sementara dulu, lalu rename untuk atomic update
            temp_path = self.index_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w') as f:
                    json.dump(self.index, f, indent=2)
                
                # Atomic replace
                shutil.move(str(temp_path), str(self.index_path))
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan cache index: {str(e)}")
                if temp_path.exists():
                    temp_path.unlink()
    
    def get_cache_key(self, file_path: str, params: Dict) -> str:
        """
        Generate cache key dari file path dan parameter.
        
        Args:
            file_path: Path file sumber
            params: Parameter preprocessing
            
        Returns:
            String cache key
        """
        # Hash file content untuk ukuran kecil
        # Untuk file besar, hash name+size+mtime untuk performa
        if os.path.getsize(file_path) < 10 * 1024 * 1024:  # < 10 MB
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        else:
            stat = os.stat(file_path)
            content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Hash parameters
        param_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        
        return f"{file_hash[:10]}_{param_hash[:10]}"
    
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
            if key not in self.index['files']:
                self._stats['misses'] += 1
                return None
            
            # Validasi entri
            file_info = self.index['files'][key]
            cache_path = self.cache_dir / f"{key}.pkl"
            
            # Cek apakah file ada di disk
            if not cache_path.exists():
                self._stats['misses'] += 1
                del self.index['files'][key]
                self.index['total_size'] -= file_info.get('size', 0)
                self.save_index()
                return None
            
            # Cek kadaluwarsa
            if self.ttl_seconds > 0:
                timestamp = datetime.fromisoformat(file_info['timestamp'])
                age = (datetime.now() - timestamp).total_seconds()
                
                if age > self.ttl_seconds:
                    self._stats['expired'] += 1
                    self._stats['misses'] += 1
                    # Tidak hapus file di sini, biarkan cleanup yang melakukannya
                    return None
                    
            # Load data
            start_time = time.time() if measure_time else 0
            
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                load_time = time.time() - start_time if measure_time else 0
                self._stats['hits'] += 1
                self._stats['saved_time'] += load_time
                
                # Update timestamp akses
                file_info['last_accessed'] = datetime.now().isoformat()
                self.save_index()
                
                return data
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat membaca cache {key}: {str(e)}")
                self._stats['misses'] += 1
                return None
    
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
            if self.index['total_size'] > 0.9 * self.max_size_bytes:
                # Cache hampir penuh, bersihkan dulu
                self.cleanup()
                
                # Jika masih penuh, tidak simpan
                if self.index['total_size'] > 0.9 * self.max_size_bytes:
                    self.logger.warning("⚠️ Cache penuh, tidak bisa menyimpan")
                    return False
            
            # Simpan data
            cache_path = self.cache_dir / f"{key}.pkl"
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Ukuran file aktual
                actual_size = cache_path.stat().st_size
                
                # Update index
                self.index['files'][key] = {
                    'size': actual_size,
                    'timestamp': datetime.now().isoformat(),
                    'last_accessed': datetime.now().isoformat()
                }
                
                self.index['total_size'] += actual_size
                self._stats['saved_bytes'] += actual_size
                
                self.save_index()
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan ke cache: {str(e)}")
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except:
                        pass
                return False
    
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
            expired_keys = []
            if self.ttl_seconds > 0:
                for key, info in self.index['files'].items():
                    timestamp = datetime.fromisoformat(info['timestamp'])
                    age = (current_time - timestamp).total_seconds()
                    
                    if age > self.ttl_seconds:
                        expired_keys.append(key)
            
            # Hapus yang kadaluwarsa
            for key in expired_keys:
                cache_path = self.cache_dir / f"{key}.pkl"
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                    
                    size = self.index['files'][key]['size']
                    del self.index['files'][key]
                    self.index['total_size'] -= size
                    
                    cleanup_stats['removed'] += 1
                    cleanup_stats['freed_bytes'] += size
                    self._stats['evictions'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error saat menghapus {key}: {str(e)}")
                    cleanup_stats['errors'] += 1
            
            # Jika hanya pembersihan expired, selesai di sini
            if expired_only and not force and self.index['total_size'] <= 0.8 * self.max_size_bytes:
                if cleanup_stats['removed'] > 0:
                    self.logger.info(
                        f"🧹 Removed {cleanup_stats['removed']} expired "
                        f"entries ({cleanup_stats['freed_bytes']/1024/1024:.1f} MB)"
                    )
                self.index['last_cleanup'] = current_time.isoformat()
                self.save_index()
                return cleanup_stats
            
            # Jika perlu pembersihan lebih lanjut
            if force or self.index['total_size'] > 0.8 * self.max_size_bytes:
                # Sort berdasarkan waktu akses
                entries = []
                for key, info in self.index['files'].items():
                    last_access = info.get('last_accessed', info['timestamp'])
                    entries.append((key, datetime.fromisoformat(last_access), info['size']))
                
                # Sort dari akses terlama
                entries.sort(key=lambda x: x[1])
                
                # Hapus sampai ukuran di bawah 70% dari max
                target_size = int(0.7 * self.max_size_bytes)
                
                for key, _, size in entries:
                    if key in expired_keys:  # Skip yang sudah dihapus
                        continue
                        
                    if self.index['total_size'] <= target_size:
                        break
                    
                    cache_path = self.cache_dir / f"{key}.pkl"
                    try:
                        if cache_path.exists():
                            cache_path.unlink()
                        
                        del self.index['files'][key]
                        self.index['total_size'] -= size
                        
                        cleanup_stats['removed'] += 1
                        cleanup_stats['freed_bytes'] += size
                        self._stats['evictions'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Error saat menghapus {key}: {str(e)}")
                        cleanup_stats['errors'] += 1
            
            # Update index
            self.index['last_cleanup'] = current_time.isoformat()
            self.save_index()
            
            # Log hasil
            if cleanup_stats['removed'] > 0:
                freed_mb = cleanup_stats['freed_bytes'] / 1024 / 1024
                percent_freed = (cleanup_stats['freed_bytes'] / max(1, self.max_size_bytes)) * 100
                
                self.logger.info(
                    f"🧹 Cache cleanup: {cleanup_stats['removed']} entries removed, "
                    f"{freed_mb:.1f} MB freed ({percent_freed:.1f}%)"
                )
            
            return cleanup_stats
    
    def clear(self) -> bool:
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
                self._init_empty_index()
                self.save_index()
                
                # Reset statistik
                self._stats = {key: 0 for key in self._stats}
                
                self.logger.success("🧹 Cache berhasil dibersihkan")
                return True
            except Exception as e:
                self.logger.error(f"❌ Gagal membersihkan cache: {str(e)}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik cache.
        
        Returns:
            Dict statistik cache
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_ratio = (self._stats['hits'] / max(1, total_requests)) * 100
            
            file_count = len(list(self.cache_dir.glob("*.pkl")))
            
            # Validasi ukuran aktual
            if file_count != len(self.index['files']):
                self.logger.warning(
                    f"⚠️ Perbedaan jumlah file cache: "
                    f"index={len(self.index['files'])}, disk={file_count}"
                )
            
            # Hitung ukuran aktual
            actual_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            if abs(actual_size - self.index['total_size']) > 1024 * 1024:  # > 1MB perbedaan
                self.logger.warning(
                    f"⚠️ Perbedaan ukuran cache: "
                    f"index={self.index['total_size']/1024/1024:.1f}MB, "
                    f"disk={actual_size/1024/1024:.1f}MB"
                )
                # Koreksi index
                self.index['total_size'] = actual_size
                self.save_index()
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_ratio': hit_ratio,
                'evictions': self._stats['evictions'],
                'expired': self._stats['expired'],
                'saved_bytes': self._stats['saved_bytes'],
                'saved_time': self._stats['saved_time'],
                'cache_size_bytes': self.index['total_size'],
                'cache_size_mb': self.index['total_size'] / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'usage_percent': (self.index['total_size'] / max(1, self.max_size_bytes)) * 100,
                'file_count': file_count,
                'last_cleanup': self.index.get('last_cleanup', 'Never')
            }
    
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
            for key in self.index['files']:
                cache_path = self.cache_dir / f"{key}.pkl"
                if not cache_path.exists():
                    missing_keys.append(key)
                    stats['missing_files'] += 1
            
            # Periksa file yang tidak terdaftar (di disk tapi tidak dalam index)
            index_keys = set(self.index['files'].keys())
            disk_files = {p.stem for p in self.cache_dir.glob("*.pkl")}
            orphaned_keys = disk_files - index_keys
            stats['orphaned_files'] = len(orphaned_keys)
            
            # Periksa file yang rusak
            corrupted_keys = []
            for key in list(self.index['files'].keys()):
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
                    size = self.index['files'][key]['size']
                    del self.index['files'][key]
                    self.index['total_size'] -= size
                    stats['fixed'] += 1
                
                # Hapus file yang tidak terdaftar
                for key in orphaned_keys:
                    try:
                        cache_path = self.cache_dir / f"{key}.pkl"
                        cache_path.unlink()
                        stats['fixed'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Gagal menghapus file orphaned {key}: {str(e)}")
                
                self.save_index()
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
    
    def __del__(self):
        """Cleanup saat objek dihapus."""
        try:
            # Simpan index saat objek dihapus
            self.save_index()
        except:
            # Diabaikan, mungkin Python sudah shutting down
            pass

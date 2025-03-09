"""
File: smartcash/handlers/preprocessing/integration/cache_adapter.py
Author: Alfrida Sabar
Deskripsi: Adapter untuk CacheManager dari utils/cache yang menyediakan
           antarmuka konsisten dengan komponen preprocessing lainnya.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import hashlib
import json

from smartcash.utils.cache import CacheManager
from smartcash.utils.logger import SmartCashLogger, get_logger


class CacheAdapter:
    """
    Adapter untuk CacheManager dari utils/cache.
    Menyediakan antarmuka yang konsisten dan memperluas fungsionalitas cache.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Inisialisasi adapter untuk cache manager.
        
        Args:
            config: Konfigurasi untuk cache
            cache_dir: Direktori cache (opsional)
            logger: Logger kustom (opsional)
            **kwargs: Parameter tambahan
        """
        self.config = config
        self.logger = logger or get_logger("CacheAdapter")
        
        # Ambil direktori cache dari config jika tidak diberikan
        self.cache_dir = cache_dir or self._get_config_value(
            'data.preprocessing.cache_dir', 
            default=".cache/smartcash"
        )
        
        # Ambil konfigurasi cache
        cache_config = self._get_config_value('data.preprocessing.cache', {})
        
        max_size_gb = cache_config.get('max_size_gb', 1.0)
        ttl_hours = cache_config.get('ttl_hours', 24)
        auto_cleanup = cache_config.get('auto_cleanup', True)
        cleanup_interval_mins = cache_config.get('cleanup_interval_mins', 30)
        
        # Buat instance cache manager
        self.cache_manager = CacheManager(
            cache_dir=self.cache_dir,
            max_size_gb=max_size_gb,
            ttl_hours=ttl_hours,
            auto_cleanup=auto_cleanup,
            cleanup_interval_mins=cleanup_interval_mins,
            logger=self.logger
        )
        
        self.logger.info(
            f"🗄️ Cache adapter diinisialisasi dengan direktori: {self.cache_dir}, "
            f"max_size: {max_size_gb}GB, ttl: {ttl_hours}h"
        )
    
    def get(self, key: str, default: Any = None, measure_time: bool = False) -> Any:
        """
        Ambil data dari cache.
        
        Args:
            key: Kunci cache
            default: Nilai default jika kunci tidak ditemukan
            measure_time: Ukur waktu loading
            
        Returns:
            Any: Data dari cache atau default
        """
        return self.cache_manager.get(key, default, measure_time=measure_time)
    
    def put(self, key: str, data: Any) -> bool:
        """
        Simpan data ke cache.
        
        Args:
            key: Kunci cache
            data: Data yang akan disimpan
            
        Returns:
            bool: True jika berhasil
        """
        return self.cache_manager.put(key, data)
    
    def exists(self, key: str) -> bool:
        """
        Cek apakah kunci ada dalam cache.
        
        Args:
            key: Kunci cache
            
        Returns:
            bool: True jika kunci ada
        """
        return self.cache_manager.exists(key)
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """
        Buat kunci cache yang konsisten dari args dan kwargs.
        
        Args:
            *args: Argumen positional
            **kwargs: Argumen keyword
            
        Returns:
            str: Kunci cache yang unik
        """
        # Gabungkan semua argumen menjadi satu string
        key_components = [str(arg) for arg in args]
        
        # Urutkan kwargs berdasarkan kunci untuk konsistensi
        for k in sorted(kwargs.keys()):
            key_components.append(f"{k}={kwargs[k]}")
            
        # Gabungkan dengan separator
        key_string = "|".join(key_components)
        
        # Buat hash
        hash_object = hashlib.md5(key_string.encode())
        hash_str = hash_object.hexdigest()
        
        return hash_str
    
    def get_preprocessing_cache_key(
        self, 
        component_name: str, 
        operation: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Buat kunci cache khusus untuk operasi preprocessing.
        
        Args:
            component_name: Nama komponen
            operation: Nama operasi
            parameters: Parameter operasi
            
        Returns:
            str: Kunci cache yang unik
        """
        # Buat prefix yang lebih mudah dibaca untuk debugging
        prefix = f"preproc_{component_name}_{operation}"
        
        # Buat json string dari parameters (diurutkan berdasarkan kunci)
        params_str = json.dumps(parameters, sort_keys=True)
        
        # Buat hash
        hash_object = hashlib.md5(params_str.encode())
        hash_str = hash_object.hexdigest()
        
        return f"{prefix}_{hash_str}"
    
    def cleanup(self, expired_only: bool = True) -> Dict[str, Any]:
        """
        Bersihkan cache.
        
        Args:
            expired_only: Hanya bersihkan entri yang kadaluwarsa
            
        Returns:
            Dict[str, Any]: Hasil pembersihan
        """
        self.logger.start("🧹 Membersihkan cache")
        result = self.cache_manager.cleanup(expired_only=expired_only)
        
        # Format hasil
        cleaned_result = {
            'status': 'success',
            'cleaned_count': result.get('cleaned_count', 0),
            'freed_space_mb': result.get('freed_space_mb', 0),
            'expired_only': expired_only
        }
        
        self.logger.success(
            f"✅ Pembersihan cache selesai: {cleaned_result['cleaned_count']} entri dibersihkan, "
            f"{cleaned_result['freed_space_mb']:.2f}MB ruang dibebaskan"
        )
        
        return cleaned_result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik cache.
        
        Returns:
            Dict[str, Any]: Statistik cache
        """
        return self.cache_manager.get_stats()
    
    def verify_integrity(self, fix: bool = True) -> Dict[str, Any]:
        """
        Verifikasi integritas cache.
        
        Args:
            fix: Perbaiki masalah yang ditemukan
            
        Returns:
            Dict[str, Any]: Hasil verifikasi
        """
        self.logger.start("🔍 Memverifikasi integritas cache")
        result = self.cache_manager.verify_integrity(fix=fix)
        
        # Format hasil
        verify_result = {
            'status': 'success',
            'integrity_issues': result.get('integrity_issues', 0),
            'fixed_issues': result.get('fixed_issues', 0),
            'fix_attempted': fix
        }
        
        self.logger.success(
            f"✅ Verifikasi integritas selesai: {verify_result['integrity_issues']} masalah ditemukan, "
            f"{verify_result['fixed_issues']} masalah diperbaiki"
        )
        
        return verify_result
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Helper untuk mendapatkan nilai dari hierarki konfigurasi.
        Mendukung dot notation (misalnya 'data.preprocessing.cache_dir').
        
        Args:
            key: Kunci konfigurasi, dapat menggunakan dot notation
            default: Nilai default jika kunci tidak ditemukan
            
        Returns:
            Any: Nilai dari konfigurasi
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
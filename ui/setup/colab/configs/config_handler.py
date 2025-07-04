"""
file_path: smartcash/ui/setup/colab/configs/config_handler.py
Deskripsi: Handler untuk manajemen konfigurasi in-memory di Colab.

Handler ini menyediakan antarmuka untuk mengelola konfigurasi lingkungan
secara in-memory tanpa melakukan operasi I/O ke file.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Union
from pathlib import Path

# Import defaults
from smartcash.ui.setup.colab.configs.defaults import DEFAULT_CONFIG

class ConfigHandler:
    """Handler untuk manajemen konfigurasi in-memory.
    
    Handler ini menyimpan konfigurasi di memori dan menyediakan
    metode untuk mengakses dan memodifikasinya.
    """
    
    def __init__(self):
        """Inisialisasi ConfigHandler dengan konfigurasi default."""
        self._config = DEFAULT_CONFIG.copy()
        self._config_path = None  # Tidak digunakan, hanya untuk kompatibilitas
    
    @property
    def config(self) -> Dict[str, Any]:
        """Mengembalikan salinan konfigurasi saat ini."""
        return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Mendapatkan nilai konfigurasi berdasarkan key.
        
        Args:
            key: Key konfigurasi (contoh: 'environment.auto_mount_drive')
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default jika tidak ditemukan
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Mengatur nilai konfigurasi.
        
        Args:
            key: Key konfigurasi (contoh: 'environment.auto_mount_drive')
            value: Nilai yang akan diset
        """
        keys = key.split('.')
        config = self._config
        
        # Navigasi ke parent dari key target
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set nilai
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Memperbarui konfigurasi dengan dictionary baru.
        
        Args:
            config_dict: Dictionary berisi pembaruan konfigurasi
        """
        self._update_dict(self._config, config_dict)
    
    def _update_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Rekursif mengupdate dictionary.
        
        Args:
            d: Dictionary target
            u: Dictionary dengan pembaruan
            
        Returns:
            Dictionary yang telah diupdate
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def reset_to_defaults(self) -> None:
        """Mengembalikan konfigurasi ke nilai default."""
        self._config = DEFAULT_CONFIG.copy()
    
    def validate_config(self) -> bool:
        """Memvalidasi konfigurasi saat ini.
        
        Returns:
            True jika konfigurasi valid, False jika tidak
        """
        # Implementasi validasi sederhana
        required_keys = [
            'environment.type',
            'environment.auto_mount_drive',
            'environment.base_path'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                return False
        
        return True

# Buat instance singleton untuk digunakan di seluruh aplikasi
config_handler = ConfigHandler()

"""
File: smartcash/common/config/manager.py
Deskripsi: Manager konfigurasi dengan dukungan YAML, environment variables, dan dependency injection
"""

import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple, List

from smartcash.common.config.singleton import Singleton
from smartcash.common.config.base_manager import BaseConfigManager
from smartcash.common.config.module_manager import ModuleConfigManager
from smartcash.common.config.drive_manager import DriveConfigManager
from smartcash.common.config.dependency_manager import DependencyManager

# Type variable untuk dependency injection
T = TypeVar('T')

class ConfigManager(DriveConfigManager, DependencyManager):
    """
    Manager untuk konfigurasi aplikasi dengan dukungan untuk loading dari file, 
    environment variable overrides, dependency injection, dan sinkronisasi dengan Google Drive
    """
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager dengan base directory, file konfigurasi utama, dan prefix environment variable
        
        Args:
            base_dir: Direktori dasar
            config_file: File konfigurasi utama
            env_prefix: Prefix untuk environment variables
        """
        if base_dir is None:
            raise ValueError("base_dir must not be None. Please provide a valid base directory for configuration.")
            
        # Initialize logger first
        from smartcash.common.logger import get_logger
        logger = get_logger("config_manager")
        # Defensive: ensure logger is not a string
        if isinstance(logger, str):
            print(f"[WARNING] Logger should not be a string! Got: {logger}. Setting logger to None.")
            logger = None
        self._logger = logger
            
        # Inisialisasi DriveConfigManager
        DriveConfigManager.__init__(self, base_dir, config_file, env_prefix)
        
        # Inisialisasi DependencyManager
        DependencyManager.__init__(self)
        
        # Simpan config_file untuk referensi
        self._config_file = config_file

    def sync_config_with_drive(self, module_name: str) -> bool:
        """
        Sinkronisasi konfigurasi modul ke Google Drive dan pastikan persistensi antar lokal dan drive.
        Args:
            module_name (str): Nama modul konfigurasi (misal: 'training_strategy')
        Returns:
            bool: True jika sinkronisasi berhasil, False jika gagal
        """
        success, message = self.sync_to_drive(module_name)
        if self._logger:
            if success:
                self._logger.info(f"✅ Sinkronisasi konfigurasi '{module_name}' ke Google Drive berhasil: {message}")
            else:
                self._logger.warning(f"⚠️ Sinkronisasi konfigurasi '{module_name}' ke Google Drive gagal: {message}")
        return success

# Singleton instance
_config_manager = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance ConfigManager (singleton).
    
    Args:
        base_dir: Direktori dasar
        config_file: File konfigurasi utama
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance singleton ConfigManager
    """
    global _config_manager
    if base_dir is None:
        raise ValueError("base_dir must not be None. Please provide a valid base directory for configuration.")
    if _config_manager is None:
        _config_manager = ConfigManager(base_dir, config_file, env_prefix)
    return _config_manager

# Tambahkan method get_instance sebagai staticmethod untuk kompatibilitas
ConfigManager.get_instance = staticmethod(get_config_manager)

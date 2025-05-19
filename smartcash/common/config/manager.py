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
        # Inisialisasi DriveConfigManager
        DriveConfigManager.__init__(self, base_dir, config_file, env_prefix)
        
        # Inisialisasi DependencyManager
        DependencyManager.__init__(self)
        
        # Simpan config_file untuk referensi
        self._config_file = config_file

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

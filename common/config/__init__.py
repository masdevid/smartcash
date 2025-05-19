"""
File: smartcash/common/config/__init__.py
Deskripsi: Package untuk manajemen konfigurasi dengan dependency injection dan sync
"""

# Import langsung dari modul manager untuk singleton pattern
from smartcash.common.config.manager import ConfigManager, get_config_manager

from smartcash.common.config.singleton import Singleton
from smartcash.common.config.base_manager import BaseConfigManager
from smartcash.common.config.module_manager import ModuleConfigManager
from smartcash.common.config.drive_manager import DriveConfigManager
from smartcash.common.config.dependency_manager import DependencyManager

from smartcash.common.config.sync import (
    sync_config_with_drive,
    sync_all_configs,
    merge_configs_smart,
    are_configs_identical
)

# Re-export fungsi utama
__all__ = [
    'ConfigManager',
    'get_config_manager',
    'Singleton',
    'BaseConfigManager',
    'ModuleConfigManager',
    'DriveConfigManager',
    'DependencyManager',
    'sync_config_with_drive',
    'sync_all_configs',
    'merge_configs_smart',
    'are_configs_identical'
]
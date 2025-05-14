"""
File: smartcash/common/config/__init__.py
Deskripsi: Package untuk manajemen konfigurasi dengan dependency injection dan sync
"""

from smartcash.common.config.manager import (
    ConfigManager, 
    get_config_manager
)

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
    'sync_config_with_drive',
    'sync_all_configs',
    'merge_configs_smart',
    'are_configs_identical'
]
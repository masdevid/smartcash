"""
File: smartcash/common/config/__init__.py
Deskripsi: Package untuk manajemen konfigurasi yang disederhanakan
"""

# Import langsung dari modul manager untuk singleton pattern
from smartcash.common.config.manager import SimpleConfigManager, get_config_manager

# Alias SimpleConfigManager sebagai ConfigManager untuk kompatibilitas
ConfigManager = SimpleConfigManager

# Re-export fungsi utama dan alias
__all__ = [
    'SimpleConfigManager',
    'ConfigManager',  # Alias untuk kompatibilitas
    'get_config_manager'
]

# Import dan expose fungsi kompatibilitas
from smartcash.common.config.compat import get_module_config
__all__.append('get_module_config')
"""
File: smartcash/common/config/compat.py
Deskripsi: Modul kompatibilitas untuk ConfigManager
"""

from smartcash.common.config.manager import ConfigManager, get_config_manager

# Tambahkan method get_instance sebagai static method ke ConfigManager
# untuk kompatibilitas dengan kode lama
def get_instance(*args, **kwargs):
    """
    Fungsi kompatibilitas untuk mendapatkan instance ConfigManager.
    Ini untuk mendukung kode lama yang menggunakan ConfigManager.get_instance()
    
    Returns:
        Instance singleton ConfigManager
    """
    return get_config_manager(*args, **kwargs)

# Tambahkan method get_instance ke class ConfigManager
ConfigManager.get_instance = staticmethod(get_instance)

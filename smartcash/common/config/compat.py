"""
File: smartcash/common/config/compat.py
Deskripsi: Fungsi-fungsi kompatibilitas untuk ConfigManager lama
"""

from smartcash.common.config import SimpleConfigManager, get_config_manager

# Tambahkan method get_instance sebagai static method ke SimpleConfigManager
def get_instance():
    """
    Fungsi kompatibilitas untuk mendapatkan instance SimpleConfigManager.
    Ini untuk mendukung kode lama yang menggunakan ConfigManager.get_instance()
    
    Returns:
        Instance singleton SimpleConfigManager
    """
    return get_config_manager()

# Tambahkan method get_instance ke class SimpleConfigManager
SimpleConfigManager.get_instance = staticmethod(get_instance)

# Fungsi kompatibilitas untuk mendukung kode lama
def get_module_config(config_name):
    """
    Fungsi kompatibilitas untuk mendapatkan konfigurasi modul.
    
    Args:
        config_name: Nama konfigurasi
        
    Returns:
        Dictionary konfigurasi
    """
    return get_instance().get_config(config_name)

# Tambahkan fungsi untuk kompatibilitas ke SimpleConfigManager
SimpleConfigManager.get_module_config = SimpleConfigManager.get_config

"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_manager.py
Deskripsi: Fungsi manajemen konfigurasi untuk hyperparameters model
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.handlers.default_config import get_default_hyperparameters_config

# Setup logger dengan level INFO untuk mengurangi log berlebihan
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def get_default_base_dir() -> str:
    """
    Dapatkan direktori base default berdasarkan environment.
    
    Returns:
        Path direktori base
    """
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_hyperparameters_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi hyperparameters dari config manager.
    
    Returns:
        Dictionary konfigurasi hyperparameters
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('hyperparameters')
        if config and 'hyperparameters' in config:
            return config
        logger.warning("⚠️ Konfigurasi hyperparameters tidak ditemukan, menggunakan default")
        return get_default_hyperparameters_config()
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()

def save_hyperparameters_config(config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi hyperparameters ke config manager.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        
    Returns:
        Status berhasil atau tidak
    """
    try:
        # Pastikan konfigurasi memiliki struktur yang benar
        if 'hyperparameters' not in config:
            config = {'hyperparameters': config}
        
        # Simpan konfigurasi
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        save_success = config_manager.save_module_config('hyperparameters', config)
        
        if not save_success:
            logger.error("❌ Gagal menyimpan konfigurasi hyperparameters")
            return False
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameters berhasil disimpan")
        return True
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi hyperparameters: {str(e)}")
        return False

def reset_hyperparameters_config() -> Dict[str, Any]:
    """
    Reset konfigurasi hyperparameters ke default dan simpan.
    
    Returns:
        Konfigurasi default yang telah disimpan
    """
    try:
        # Dapatkan konfigurasi default
        default_config = get_default_hyperparameters_config()
        
        # Simpan konfigurasi default
        save_success = save_hyperparameters_config(default_config)
        
        if not save_success:
            logger.error("❌ Gagal menyimpan konfigurasi default hyperparameters")
            return default_config
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameters berhasil direset ke default")
        return default_config
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat reset konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()
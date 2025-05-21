"""
File: smartcash/ui/training_config/training_strategy/handlers/config_loader.py
Deskripsi: Fungsi untuk memuat konfigurasi training strategy
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

from smartcash.ui.training_config.training_strategy.handlers.default_config import (
    get_default_training_strategy_config,
    get_default_base_dir
)

logger = get_logger(__name__)

def get_training_strategy_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi training strategy dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi training strategy
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('training_strategy')
        if config:
            # Pastikan konfigurasi memiliki struktur yang diharapkan
            if 'training_strategy' not in config:
                config = {'training_strategy': config}
            # Pastikan semua key diperlukan ada - tambahkan default jika tidak
            default_config = get_default_training_strategy_config()['training_strategy']
            for key in default_config:
                if key not in config['training_strategy']:
                    config['training_strategy'][key] = default_config[key]
            return config
        logger.warning(f"{ICONS.get('warning', '⚠️')} Konfigurasi training strategy tidak ditemukan, menggunakan default")
        return get_default_training_strategy_config()
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengambil konfigurasi training strategy: {str(e)}")
        return get_default_training_strategy_config()

def save_training_strategy_config(config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi training strategy ke config manager.
    
    Args:
        config: Konfigurasi yang akan disimpan
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # Pastikan konfigurasi memiliki struktur yang benar
        if 'training_strategy' not in config:
            config = {'training_strategy': config}
            
        # Simpan konfigurasi
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        result = config_manager.save_module_config('training_strategy', config)
        
        if result:
            logger.info(f"{ICONS.get('success', '✅')} Konfigurasi training strategy berhasil disimpan")
        else:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menyimpan konfigurasi training strategy")
            
        return result
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi training strategy: {str(e)}")
        return False
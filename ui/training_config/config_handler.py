"""
File: smartcash/ui/training_config/config_handler.py
Deskripsi: Handler untuk konfigurasi training yang menggunakan SimpleConfigManager
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import os
import yaml
import logging
from pathlib import Path

from smartcash.common.config import get_config_manager, SimpleConfigManager
from smartcash.common.logger import get_logger

logger = get_logger()

def load_training_config(config_name: str = "training") -> Dict[str, Any]:
    """
    Memuat konfigurasi training dari SimpleConfigManager.
    
    Args:
        config_name: Nama konfigurasi (default: "training")
        
    Returns:
        Dict[str, Any]: Konfigurasi training
    """
    try:
        # Gunakan SimpleConfigManager untuk load konfigurasi
        config_manager = get_config_manager()
        config = config_manager.get_module_config(config_name)
        
        if config:
            logger.info(f"✅ Konfigurasi training berhasil dimuat dari SimpleConfigManager")
            return config
        else:
            logger.warning(f"⚠️ Konfigurasi training tidak ditemukan di SimpleConfigManager, menggunakan default kosong")
            return {}
    except Exception as e:
        logger.error(f"❌ Error saat memuat konfigurasi training: {str(e)}")
        return {}

def save_training_config(config: Dict[str, Any], config_name: str = "training") -> bool:
    """
    Menyimpan konfigurasi training menggunakan SimpleConfigManager.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_name: Nama konfigurasi (default: "training")
        
    Returns:
        bool: True jika berhasil disimpan, False jika tidak
    """
    try:
        # Gunakan SimpleConfigManager untuk save konfigurasi
        config_manager = get_config_manager()
        result = config_manager.save_module_config(config_name, config)
        
        if result:
            logger.info(f"✅ Konfigurasi training berhasil disimpan")
            return True
        else:
            logger.warning(f"⚠️ Gagal menyimpan konfigurasi training")
            return False
    except Exception as e:
        logger.error(f"❌ Error saat menyimpan konfigurasi training: {str(e)}")
        return False

def update_training_config(update_dict: Dict[str, Any], config_name: str = "training") -> bool:
    """
    Memperbarui konfigurasi training yang ada dengan nilai baru.
    
    Args:
        update_dict: Nilai-nilai yang akan diperbarui
        config_name: Nama konfigurasi (default: "training")
        
    Returns:
        bool: True jika berhasil diperbarui, False jika tidak
    """
    try:
        # Load konfigurasi saat ini
        config = load_training_config(config_name)
        
        # Update dengan nilai baru
        config.update(update_dict)
        
        # Simpan kembali konfigurasi yang sudah diupdate
        return save_training_config(config, config_name)
    except Exception as e:
        logger.error(f"❌ Error saat memperbarui konfigurasi training: {str(e)}")
        return False

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Memperbarui UI berdasarkan konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi sumber
    """
    try:
        # Update widget UI berdasarkan config yang tersedia
        for key, value in config.items():
            # Cari komponen UI yang sesuai dengan key
            if key in ui_components and hasattr(ui_components[key], 'value'):
                try:
                    ui_components[key].value = value
                except Exception as e:
                    logger.debug(f"⚠️ Tidak bisa mengatur nilai '{key}': {str(e)}")
            # Cek nested dictionary seperti 'hyperparameters' atau 'backbone'
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    combined_key = f"{key}_{nested_key}"
                    if combined_key in ui_components and hasattr(ui_components[combined_key], 'value'):
                        try:
                            ui_components[combined_key].value = nested_value
                        except Exception as e:
                            logger.debug(f"⚠️ Tidak bisa mengatur nilai '{combined_key}': {str(e)}")
        
        logger.info(f"✅ UI berhasil diperbarui dari konfigurasi")
    except Exception as e:
        logger.error(f"❌ Error saat memperbarui UI dari konfigurasi: {str(e)}")

def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict[str, Any]: Konfigurasi dari UI
    """
    config = {}
    
    # Daftar kunci yang akan diambil dari UI
    # Catatan: Ini adalah contoh umum, sesuaikan dengan kebutuhan spesifik
    config_keys = [
        'backbone_type', 'epochs', 'batch_size', 'learning_rate',
        'optimizer', 'loss_function', 'augmentation', 'validation_split'
    ]
    
    # Ekstrak nilai dari UI components
    for key in config_keys:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            config[key] = ui_components[key].value
    
    # Grup hyperparameters berdasarkan prefix (jika ada)
    hyperparams = {}
    for key in ui_components:
        if key.startswith('hyperparams_') and hasattr(ui_components[key], 'value'):
            param_name = key.replace('hyperparams_', '')
            hyperparams[param_name] = ui_components[key].value
    
    # Tambahkan hyperparameters ke config jika ada
    if hyperparams:
        config['hyperparameters'] = hyperparams
    
    return config 
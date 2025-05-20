"""
File: smartcash/ui/dataset/split/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk split dataset
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def get_default_base_dir():
    """Dapatkan direktori base default."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_default_split_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "split": {
            "enabled": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "stratify": True
        }
    }

def load_config() -> Dict[str, Any]:
    """
    Load konfigurasi split dataset dari config manager.
    
    Returns:
        Dictionary konfigurasi split
    """
    try:
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        config = config_manager.get_module_config('split', {})
        
        # Pastikan config memiliki struktur yang benar
        if not config or 'split' not in config:
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan konfigurasi default untuk split dataset")
            default_config = get_default_split_config()
            # Simpan default config ke file
            config_manager.save_module_config('split', default_config)
            return default_config
            
        return config
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat load konfigurasi split: {str(e)}")
        return get_default_split_config()

def save_config(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> None:
    """
    Simpan konfigurasi split dataset ke config manager.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        ui_components: Dictionary komponen UI (opsional)
    """
    try:
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        config_manager.save_module_config('split', config)
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi split berhasil disimpan")
        
        # Log ke UI jika ui_components tersedia
        if ui_components and 'logger' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_success
            log_sync_success(ui_components, "Konfigurasi split berhasil disimpan")
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi split: {str(e)}")
        
        # Log ke UI jika ui_components tersedia
        if ui_components and 'logger' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_error
            log_sync_error(ui_components, f"Error saat menyimpan konfigurasi split: {str(e)}")

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Import sync logger
        from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_info, log_sync_success, log_sync_error
        
        # Log sync start
        log_sync_info(ui_components, "Memulai sinkronisasi UI dari konfigurasi...")
        
        # Get config if not provided
        if config is None:
            config = load_config()
            
        # Pastikan config memiliki struktur yang benar
        if not config or 'split' not in config:
            log_sync_info(ui_components, "Menggunakan konfigurasi default untuk split dataset")
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan konfigurasi default untuk split dataset")
            config = get_default_split_config()
            
        split_config = config['split']
        
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = split_config.get('enabled', True)
            
        if 'train_ratio_slider' in ui_components:
            ui_components['train_ratio_slider'].value = split_config.get('train_ratio', 0.7)
            
        if 'val_ratio_slider' in ui_components:
            ui_components['val_ratio_slider'].value = split_config.get('val_ratio', 0.15)
            
        if 'test_ratio_slider' in ui_components:
            ui_components['test_ratio_slider'].value = split_config.get('test_ratio', 0.15)
            
        if 'random_seed_input' in ui_components:
            ui_components['random_seed_input'].value = split_config.get('random_seed', 42)
            
        if 'stratify_checkbox' in ui_components:
            ui_components['stratify_checkbox'].value = split_config.get('stratify', True)
            
        log_sync_success(ui_components, "UI berhasil diupdate dari konfigurasi split")
        logger.info(f"{ICONS.get('success', '✅')} UI berhasil diupdate dari konfigurasi split")
        
    except Exception as e:
        # Log error
        if 'logger' in ui_components:
            log_sync_error(ui_components, f"Error saat mengupdate UI dari konfigurasi: {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat mengupdate UI dari konfigurasi: {str(e)}")
        
        # Jika terjadi error, gunakan konfigurasi default
        default_config = get_default_split_config()
        update_ui_from_config(ui_components, default_config)

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Import sync logger
        from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_info, log_sync_success, log_sync_error
        
        # Log sync start
        log_sync_info(ui_components, "Memulai sinkronisasi konfigurasi dari UI...")
        
        # Get current config
        config = load_config()
        
        # Update config from UI
        if 'enabled_checkbox' in ui_components:
            config['split']['enabled'] = ui_components['enabled_checkbox'].value
            
        if 'train_ratio_slider' in ui_components:
            config['split']['train_ratio'] = ui_components['train_ratio_slider'].value
            
        if 'val_ratio_slider' in ui_components:
            config['split']['val_ratio'] = ui_components['val_ratio_slider'].value
            
        if 'test_ratio_slider' in ui_components:
            config['split']['test_ratio'] = ui_components['test_ratio_slider'].value
            
        if 'random_seed_input' in ui_components:
            config['split']['random_seed'] = ui_components['random_seed_input'].value
            
        if 'stratify_checkbox' in ui_components:
            config['split']['stratify'] = ui_components['stratify_checkbox'].value
        
        # Save config
        save_config(config, ui_components)
        
        log_sync_success(ui_components, "Konfigurasi berhasil diupdate dari UI")
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi berhasil diupdate dari UI")
        return config
        
    except Exception as e:
        # Log error
        if 'logger' in ui_components:
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_error
            log_sync_error(ui_components, f"Error saat update config dari UI: {str(e)}")
        logger.error(f"{ICONS.get('error', '❌')} Error saat update config dari UI: {str(e)}")
        return load_config()

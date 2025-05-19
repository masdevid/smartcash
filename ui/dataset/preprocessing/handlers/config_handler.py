"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
import yaml
import copy
from pathlib import Path
from IPython.display import display
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_PREPROCESSED_DIR, DEFAULT_INVALID_DIR, DEFAULT_IMG_SIZE
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def get_default_preprocessing_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk preprocessing dataset.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "preprocessing": {
            "enabled": True,
            "output_dir": DEFAULT_PREPROCESSED_DIR,
            "img_size": DEFAULT_IMG_SIZE,
            "normalization": {
                "enabled": True,
                "preserve_aspect_ratio": True
            },
            "validate": {
                "enabled": True,
                "fix_issues": True,
                "move_invalid": True,
                "invalid_dir": DEFAULT_INVALID_DIR
            },
            "splits": DEFAULT_SPLITS,
            "num_workers": 4
        },
        "data": {
            "dir": "data"
        }
    }

def get_preprocessing_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi preprocessing dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi preprocessing
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_module_config('preprocessing')
        
        if config:
            return config
            
        # Jika tidak ada config, gunakan default
        logger.warning("⚠️ Konfigurasi preprocessing tidak ditemukan, menggunakan default")
        return get_default_preprocessing_config()
        
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi preprocessing: {str(e)}")
        return get_default_preprocessing_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi preprocessing dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get current config
        config = config_manager.get_module_config('preprocessing') or get_default_preprocessing_config()
        
        # Update config from UI
        if 'enabled_checkbox' in ui_components:
            config['preprocessing']['enabled'] = ui_components['enabled_checkbox'].value
            
        if 'output_dir' in ui_components:
            config['preprocessing']['output_dir'] = ui_components['output_dir'].value
            
        if 'img_size_slider' in ui_components:
            config['preprocessing']['img_size'] = ui_components['img_size_slider'].value
            
        if 'normalize_checkbox' in ui_components:
            config['preprocessing']['normalization']['enabled'] = ui_components['normalize_checkbox'].value
            
        if 'preserve_aspect_ratio_checkbox' in ui_components:
            config['preprocessing']['normalization']['preserve_aspect_ratio'] = ui_components['preserve_aspect_ratio_checkbox'].value
            
        if 'validate_checkbox' in ui_components:
            config['preprocessing']['validate']['enabled'] = ui_components['validate_checkbox'].value
            
        if 'fix_issues_checkbox' in ui_components:
            config['preprocessing']['validate']['fix_issues'] = ui_components['fix_issues_checkbox'].value
            
        if 'move_invalid_checkbox' in ui_components:
            config['preprocessing']['validate']['move_invalid'] = ui_components['move_invalid_checkbox'].value
            
        if 'invalid_dir' in ui_components:
            config['preprocessing']['validate']['invalid_dir'] = ui_components['invalid_dir'].value
            
        if 'num_workers_slider' in ui_components:
            config['preprocessing']['num_workers'] = ui_components['num_workers_slider'].value
            
        # Save config
        config_manager.save_module_config('preprocessing', config)
        
        logger.info("✅ Konfigurasi preprocessing berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi preprocessing: {str(e)}")
        return get_default_preprocessing_config()

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            config = get_preprocessing_config(ui_components)
            
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = config['preprocessing']['enabled']
            
        if 'output_dir' in ui_components:
            ui_components['output_dir'].value = config['preprocessing']['output_dir']
            
        if 'img_size_slider' in ui_components:
            ui_components['img_size_slider'].value = config['preprocessing']['img_size']
            
        if 'normalize_checkbox' in ui_components:
            ui_components['normalize_checkbox'].value = config['preprocessing']['normalization']['enabled']
            
        if 'preserve_aspect_ratio_checkbox' in ui_components:
            ui_components['preserve_aspect_ratio_checkbox'].value = config['preprocessing']['normalization']['preserve_aspect_ratio']
            
        if 'validate_checkbox' in ui_components:
            ui_components['validate_checkbox'].value = config['preprocessing']['validate']['enabled']
            
        if 'fix_issues_checkbox' in ui_components:
            ui_components['fix_issues_checkbox'].value = config['preprocessing']['validate']['fix_issues']
            
        if 'move_invalid_checkbox' in ui_components:
            ui_components['move_invalid_checkbox'].value = config['preprocessing']['validate']['move_invalid']
            
        if 'invalid_dir' in ui_components:
            ui_components['invalid_dir'].value = config['preprocessing']['validate']['invalid_dir']
            
        if 'num_workers_slider' in ui_components:
            ui_components['num_workers_slider'].value = config['preprocessing']['num_workers']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi preprocessing")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")
"""
File: smartcash/ui/dataset/augmentation/utils/config_utils.py
Deskripsi: Utility untuk manajemen konfigurasi pada modul augmentasi dataset
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.common.environment import get_default_base_dir

logger = get_logger()

# Konstanta untuk konfigurasi default
DEFAULT_CONFIG_PATH = "configs/augmentation_config.yaml"
DEFAULT_AUGMENTATION_DIR = "data/augmented"

def get_module_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi augmentasi dari SimpleConfigManager atau file.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi augmentasi
    """
    # Dapatkan config manager
    try:
        config_manager = get_config_manager()
        
        # Coba dapatkan konfigurasi dengan metode get_module_config
        config = config_manager.get_module_config('augmentation')
        if config and isinstance(config, dict):
            logger.debug("✅ Konfigurasi augmentasi berhasil dimuat dari SimpleConfigManager")
            return config
    except Exception as e:
        logger.debug(f"Tidak dapat memuat dari SimpleConfigManager: {str(e)}")
    
    # Jika tidak dapat memuat dari ConfigManager, buat default dan simpan ke ConfigManager
    default_config = create_default_config()
    
    # Coba simpan default config ke ConfigManager
    try:
        config_manager = get_config_manager()
        config_manager.save_module_config('augmentation', default_config)
        logger.debug("✅ Default konfigurasi augmentasi berhasil disimpan ke SimpleConfigManager")
    except Exception as e:
        logger.debug(f"Tidak dapat menyimpan default config ke SimpleConfigManager: {str(e)}")
        
        # Fallback: Simpan ke file
        try:
            os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.debug(f"✅ Default konfigurasi augmentasi berhasil disimpan ke {DEFAULT_CONFIG_PATH}")
        except Exception as file_e:
            logger.warning(f"⚠️ Gagal menyimpan default config ke file: {str(file_e)}")
    
    return default_config

def save_module_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi augmentasi ke SimpleConfigManager dan file.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Dictionary konfigurasi augmentasi
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Simpan ke SimpleConfigManager menggunakan metode save_module_config
        success = config_manager.save_module_config('augmentation', config)
        
        if success:
            logger.info("✅ Konfigurasi augmentasi berhasil disimpan ke SimpleConfigManager")
            
            # Simpan juga ke file untuk kompatibilitas
            try:
                os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
                with open(DEFAULT_CONFIG_PATH, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.debug(f"✅ Konfigurasi augmentasi berhasil disimpan ke {DEFAULT_CONFIG_PATH}")
            except Exception as file_e:
                logger.debug(f"⚠️ Gagal menyimpan config ke file: {str(file_e)}")
                # Tidak menganggap ini sebagai kegagalan karena sudah berhasil disimpan ke ConfigManager
            
            return True
        else:
            logger.error("❌ SimpleConfigManager gagal menyimpan konfigurasi")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        
        # Fallback: Coba simpan ke file sebagai alternatif
        try:
            os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"✅ Konfigurasi berhasil disimpan ke file: {DEFAULT_CONFIG_PATH}")
            return True
        except Exception:
            return False

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang diupdate
    """
    config = ui_components.get('config', {})
    if 'augmentation' not in config:
        config['augmentation'] = {}
    
    # Dapatkan parameter dari augmentation_options
    aug_options = ui_components.get('augmentation_options')
    if aug_options and hasattr(aug_options, 'children') and len(aug_options.children) >= 5:
        # Ekstrak nilai dari UI
        rotation_range = aug_options.children[0].value
        width_shift = aug_options.children[1].value
        height_shift = aug_options.children[2].value
        zoom_range = aug_options.children[3].value
        horizontal_flip = aug_options.children[4].value
        
        # Update config
        config['augmentation'].update({
            'rotation_range': rotation_range,
            'width_shift_range': width_shift,
            'height_shift_range': height_shift,
            'zoom_range': zoom_range,
            'horizontal_flip': horizontal_flip
        })
    
    # Dapatkan parameter dari advanced_options
    adv_options = ui_components.get('advanced_options')
    if adv_options and hasattr(adv_options, 'children') and len(adv_options.children) >= 3:
        # Ekstrak nilai dari UI
        brightness_range = adv_options.children[0].value
        shear_range = adv_options.children[1].value
        vertical_flip = adv_options.children[2].value
        
        # Update config
        config['augmentation'].update({
            'brightness_range': brightness_range,
            'shear_range': shear_range,
            'vertical_flip': vertical_flip
        })
    
    # Dapatkan parameter dari split_selector
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value'):
        config['augmentation']['split'] = split_selector.value
    
    # Update config di ui_components
    ui_components['config'] = config
    
    return config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI components dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Dictionary konfigurasi
        
    Returns:
        Dictionary UI components yang diupdate
    """
    # Dapatkan konfigurasi augmentasi
    aug_config = config.get('augmentation', {})
    
    # Update augmentation_options
    aug_options = ui_components.get('augmentation_options')
    if aug_options and hasattr(aug_options, 'children') and len(aug_options.children) >= 5:
        # Update nilai UI
        aug_options.children[0].value = aug_config.get('rotation_range', 20)
        aug_options.children[1].value = aug_config.get('width_shift_range', 0.2)
        aug_options.children[2].value = aug_config.get('height_shift_range', 0.2)
        aug_options.children[3].value = aug_config.get('zoom_range', 0.2)
        aug_options.children[4].value = aug_config.get('horizontal_flip', True)
    
    # Update advanced_options
    adv_options = ui_components.get('advanced_options')
    if adv_options and hasattr(adv_options, 'children') and len(adv_options.children) >= 3:
        # Update nilai UI
        adv_options.children[0].value = aug_config.get('brightness_range', 0.2)
        adv_options.children[1].value = aug_config.get('shear_range', 0.2)
        adv_options.children[2].value = aug_config.get('vertical_flip', False)
    
    # Update split_selector
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value'):
        split_selector.value = aug_config.get('split', 'Train Only')
    
    # Update config di ui_components
    ui_components['config'] = config
    
    return ui_components

def create_default_config() -> Dict[str, Any]:
    """
    Buat konfigurasi default untuk augmentasi.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        'augmentation': {
            'enabled': True,
            'output_dir': DEFAULT_AUGMENTATION_DIR,
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'brightness_range': 0.2,
            'shear_range': 0.2,
            'vertical_flip': False,
            'split': 'Train Only',
            'augmentation_factor': 2
        }
    }

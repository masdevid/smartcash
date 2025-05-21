"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler untuk konfigurasi preprocessing dataset
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

# Import utils dari preprocessing module
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.notification_manager import notify_config, PREPROCESSING_LOGGER_NAMESPACE

# Setup logger
logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)

def get_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get konfigurasi preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi preprocessing dataset
    """
    try:
        # Get config manager
        config_manager = get_config_manager()
        
        # Get config
        config = config_manager.get_config('dataset')
        
        # Ensure config structure
        if not config:
            config = get_default_preprocessing_config()
        elif 'preprocessing' not in config:
            config['preprocessing'] = get_default_preprocessing_config()['preprocessing']
            
        # Log berhasil get config
        log_message(ui_components, "Konfigurasi preprocessing berhasil dimuat", "debug", "📄")
            
        return config
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat get preprocessing config: {str(e)}", "error", "❌")
        return get_default_preprocessing_config()

def get_default_preprocessing_config() -> Dict[str, Any]:
    """
    Get konfigurasi default preprocessing dataset.
    
    Returns:
        Dictionary konfigurasi default preprocessing dataset
    """
    return {
        'preprocessing': {
            'img_size': 640,
            'normalization': {
                'enabled': True,
                'preserve_aspect_ratio': True
            },
            'enabled': True,
            'num_workers': 4,
            'validate': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
                'invalid_dir': 'invalid'
            },
            'splits': ['train', 'valid', 'test']
        }
    }

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        # Get current config
        config = get_preprocessing_config(ui_components)
        
        # Update preprocessing config
        if 'preprocess_options' in ui_components:
            preproc_options = ui_components['preprocess_options']
            if hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
                # Update image size
                config['preprocessing']['img_size'] = preproc_options.children[0].value
                
                # Update normalization options
                config['preprocessing']['normalization']['enabled'] = preproc_options.children[1].value
                config['preprocessing']['normalization']['preserve_aspect_ratio'] = preproc_options.children[2].value
                
                # Update cache dan workers
                config['preprocessing']['enabled'] = preproc_options.children[3].value
                config['preprocessing']['num_workers'] = preproc_options.children[4].value
        
        # Update validation options
        if 'validation_options' in ui_components:
            validation_options = ui_components['validation_options']
            if hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
                config['preprocessing']['validate']['enabled'] = validation_options.children[0].value
                config['preprocessing']['validate']['fix_issues'] = validation_options.children[1].value
                config['preprocessing']['validate']['move_invalid'] = validation_options.children[2].value
                config['preprocessing']['validate']['invalid_dir'] = validation_options.children[3].value
        
        # Update split selector
        if 'split_selector' in ui_components:
            split_selector = ui_components['split_selector']
            if hasattr(split_selector, 'value'):
                split_map = {
                    'All Splits': ['train', 'valid', 'test'],
                    'Train Only': ['train'],
                    'Validation Only': ['valid'],
                    'Test Only': ['test']
                }
                config['preprocessing']['splits'] = split_map.get(split_selector.value, ['train', 'valid', 'test'])
            
        # Save config
        config_manager = get_config_manager()
        config_manager.save_config(config, 'dataset')
        
        # Log berhasil update config
        log_message(ui_components, "Konfigurasi preprocessing berhasil diupdate dari UI", "success", "✅")
        
        # Notifikasi melalui observer
        notify_config(ui_components, "updated", config)
        
        return config
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat update config dari UI: {str(e)}", "error", "❌")
        return get_preprocessing_config(ui_components)

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    try:
        # Ensure config structure
        if not config:
            config = get_default_preprocessing_config()
        elif 'preprocessing' not in config:
            config['preprocessing'] = get_default_preprocessing_config()['preprocessing']
            
        # Update UI components
        if 'preprocess_options' in ui_components:
            preproc_options = ui_components['preprocess_options']
            if hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
                # Update image size
                preproc_options.children[0].value = config['preprocessing']['img_size']
                
                # Update normalization options
                preproc_options.children[1].value = config['preprocessing']['normalization']['enabled']
                preproc_options.children[2].value = config['preprocessing']['normalization']['preserve_aspect_ratio']
                
                # Update cache dan workers
                preproc_options.children[3].value = config['preprocessing']['enabled']
                preproc_options.children[4].value = config['preprocessing']['num_workers']
        
        # Update validation options
        if 'validation_options' in ui_components:
            validation_options = ui_components['validation_options']
            if hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
                validation_options.children[0].value = config['preprocessing']['validate']['enabled']
                validation_options.children[1].value = config['preprocessing']['validate']['fix_issues']
                validation_options.children[2].value = config['preprocessing']['validate']['move_invalid']
                validation_options.children[3].value = config['preprocessing']['validate']['invalid_dir']
        
        # Update split selector
        if 'split_selector' in ui_components:
            split_selector = ui_components['split_selector']
            if hasattr(split_selector, 'value'):
                split_map = {
                    'All Splits': ['train', 'valid', 'test'],
                    'Train Only': ['train'],
                    'Validation Only': ['valid'],
                    'Test Only': ['test']
                }
                for key, value in split_map.items():
                    if value == config['preprocessing']['splits']:
                        split_selector.value = key
                        break
            
        # Log berhasil update UI
        log_message(ui_components, "UI preprocessing berhasil diupdate dari konfigurasi", "success", "✅")
        
        # Notifikasi melalui observer
        notify_config(ui_components, "loaded", config)
        
        return ui_components
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat update UI dari config: {str(e)}", "error", "❌")
        return ui_components

def save_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simpan konfigurasi preprocessing ke file.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang disimpan
    """
    try:
        # Update config dari UI
        config = update_config_from_ui(ui_components)
        
        # Save config
        config_manager = get_config_manager()
        config_manager.save_config(config, 'dataset')
        
        # Log berhasil save config
        log_message(ui_components, "Konfigurasi preprocessing berhasil disimpan", "success", "💾")
        
        # Notifikasi melalui observer
        notify_config(ui_components, "saved", config)
        
        return config
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat simpan config: {str(e)}", "error", "❌")
        return get_preprocessing_config(ui_components)

def reset_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset konfigurasi preprocessing ke default.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi default
    """
    try:
        # Get default config
        config = get_default_preprocessing_config()
        
        # Update UI dari config
        ui_components = update_ui_from_config(ui_components, config)
        
        # Save config
        config_manager = get_config_manager()
        config_manager.save_config(config, 'dataset')
        
        # Log berhasil reset config
        log_message(ui_components, "Konfigurasi preprocessing berhasil direset ke default", "success", "🔄")
        
        # Notifikasi melalui observer
        notify_config(ui_components, "reset", config)
        
        return config
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat reset config: {str(e)}", "error", "❌")
        return get_preprocessing_config(ui_components)

def setup_preprocessing_config_handler(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None, env: Any = None) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi (opsional)
        env: Environment manager (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Load config jika tidak diberikan
        if not config:
            config = get_preprocessing_config(ui_components)
        
        # Update UI dari konfigurasi
        ui_components = update_ui_from_config(ui_components, config)
        
        # Tambahkan fungsi ke ui_components
        ui_components['get_preprocessing_config'] = get_preprocessing_config
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['save_preprocessing_config'] = save_preprocessing_config
        ui_components['reset_preprocessing_config'] = reset_preprocessing_config
        
        # Tambahkan event handler untuk button
        if 'save_button' in ui_components and hasattr(ui_components['save_button'], 'on_click'):
            ui_components['save_button'].on_click(lambda b: save_preprocessing_config(ui_components))
        
        if 'reset_button' in ui_components and hasattr(ui_components['reset_button'], 'on_click'):
            ui_components['reset_button'].on_click(lambda b: reset_preprocessing_config(ui_components))
        
        # Log berhasil setup config handler
        log_message(ui_components, "Config handler preprocessing berhasil disetup", "debug", "✅")
        
        return ui_components
        
    except Exception as e:
        # Log error
        log_message(ui_components, f"Error saat setup config handler: {str(e)}", "error", "❌")
        return ui_components
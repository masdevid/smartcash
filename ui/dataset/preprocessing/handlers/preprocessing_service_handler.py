"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_service_handler.py
Deskripsi: Handler untuk interaksi dengan service preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger("preprocessing_service")

def get_dataset_manager(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None, custom_logger=None) -> Any:
    """
    Dapatkan instance dataset manager untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)
        custom_logger: Logger kustom (opsional)
        
    Returns:
        Instance dataset manager atau None jika gagal
    """
    logger = custom_logger or ui_components.get('logger', get_logger("preprocessing_service"))
    
    try:
        # Import dataset manager
        from smartcash.dataset.services.dataset_manager import DatasetManager
        
        # Dapatkan path dari UI components
        data_dir = ui_components.get('data_dir', 'data')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Pastikan path dalam bentuk string
        data_dir_str = str(data_dir) if isinstance(data_dir, Path) else data_dir
        preprocessed_dir_str = str(preprocessed_dir) if isinstance(preprocessed_dir, Path) else preprocessed_dir
        
        # Buat instance dataset manager
        dataset_manager = DatasetManager(
            dataset_dir=data_dir_str,
            preprocessed_dir=preprocessed_dir_str,
            logger=logger
        )
        
        # Update konfigurasi jika tersedia
        if config and hasattr(dataset_manager, 'config'):
            # Update konfigurasi dataset
            if 'data' in config:
                dataset_manager.config.update({
                    'dataset_dir': data_dir_str,
                    'preprocessed_dir': preprocessed_dir_str
                })
            
            # Update konfigurasi preprocessing
            if 'preprocessing' in config:
                dataset_manager.config['preprocessing'] = config.get('preprocessing', {})
        
        # Log info
        logger.info(f"{ICONS['info']} Dataset manager berhasil dibuat dengan direktori:")
        logger.info(f"  - Data: {data_dir_str}")
        logger.info(f"  - Preprocessed: {preprocessed_dir_str}")
        
        return dataset_manager
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat membuat dataset manager: {str(e)}")
        return None

def extract_preprocess_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak parameter preprocessing dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary parameter preprocessing
    """
    # Dapatkan opsi preprocessing dari UI
    preprocess_options = ui_components.get('preprocess_options', {})
    
    # Inisialisasi parameter default
    params = {
        'force_reprocess': True,
        'normalize': True,
        'preserve_aspect_ratio': True,
        'img_size': 640,
        'cache_images': True,
        'num_workers': 4
    }
    
    # Jika preprocess_options tersedia, update params
    if preprocess_options and hasattr(preprocess_options, 'children'):
        if len(preprocess_options.children) >= 5:
            # Ekstrak nilai dari UI components
            params.update({
                'img_size': preprocess_options.children[0].value,
                'normalize': preprocess_options.children[1].value,
                'preserve_aspect_ratio': preprocess_options.children[2].value,
                'cache_images': preprocess_options.children[3].value,
                'num_workers': preprocess_options.children[4].value
            })
    
    # Dapatkan opsi validasi dari UI
    validation_options = ui_components.get('validation_options', {})
    
    # Inisialisasi parameter validasi default
    validation_params = {
        'validate': True,
        'fix_issues': True,
        'move_invalid': True,
        'invalid_dir': 'data/invalid'
    }
    
    # Jika validation_options tersedia, update validation_params
    if validation_options and hasattr(validation_options, 'children'):
        if len(validation_options.children) >= 4:
            # Ekstrak nilai dari UI components
            validation_params.update({
                'validate': validation_options.children[0].value,
                'fix_issues': validation_options.children[1].value,
                'move_invalid': validation_options.children[2].value,
                'invalid_dir': validation_options.children[3].value
            })
    
    # Gabungkan params dan validation_params
    params.update(validation_params)
    
    return params

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Pastikan persistensi UI components dengan ConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Import ConfigManager
        from smartcash.common.config_manager import ConfigManager
        
        # Dapatkan instance ConfigManager
        config_manager = ConfigManager.get_instance()
        
        # Register UI components untuk persistensi
        config_manager.register_ui_components('preprocessing', ui_components)
        
        # Dapatkan konfigurasi preprocessing
        if not config:
            config = config_manager.get_module_config('preprocessing')
        
        # Update UI dari konfigurasi
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_ui_from_config
        ui_components = update_ui_from_config(ui_components, config)
        
        # Log info
        logger.info(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat memastikan persistensi UI: {str(e)}")
        return ui_components

def get_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi preprocessing dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi preprocessing
    """
    # Ekstrak parameter preprocessing dari UI
    preprocess_params = extract_preprocess_params(ui_components)
    
    # Dapatkan split dari UI
    split_option = ui_components.get('split_selector', {}).value if 'split_selector' in ui_components else 'All Splits'
    split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
    split = split_map.get(split_option)
    
    # Tambahkan split ke parameter
    preprocess_params['split'] = split
    
    return preprocess_params

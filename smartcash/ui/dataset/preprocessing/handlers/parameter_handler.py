"""
File: smartcash/ui/dataset/preprocessing/handlers/parameter_handler.py
Deskripsi: Handler untuk ekstraksi dan validasi parameter preprocessing
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger("preprocessing_params")

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

def validate_preprocessing_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi parameter preprocessing untuk memastikan nilai yang valid.
    
    Args:
        params: Dictionary parameter preprocessing
        
    Returns:
        Dictionary parameter preprocessing yang divalidasi
    """
    # Validasi img_size
    if 'img_size' not in params or not params['img_size'] or params['img_size'] < 32:
        logger.warning(f"{ICONS['warning']} Parameter img_size tidak valid, menggunakan default 640")
        params['img_size'] = 640
    
    # Validasi num_workers
    if 'num_workers' not in params or not isinstance(params['num_workers'], int) or params['num_workers'] < 1:
        logger.warning(f"{ICONS['warning']} Parameter num_workers tidak valid, menggunakan default 4")
        params['num_workers'] = 4
    
    # Validasi boolean parameters
    bool_params = ['normalize', 'preserve_aspect_ratio', 'cache_images', 'validate', 'fix_issues', 'move_invalid']
    for param in bool_params:
        if param not in params or not isinstance(params[param], bool):
            logger.warning(f"{ICONS['warning']} Parameter {param} tidak valid, menggunakan default True")
            params[param] = True
    
    # Validasi invalid_dir
    if 'invalid_dir' not in params or not params['invalid_dir']:
        logger.warning(f"{ICONS['warning']} Parameter invalid_dir tidak valid, menggunakan default 'data/invalid'")
        params['invalid_dir'] = 'data/invalid'
    
    return params

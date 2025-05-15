"""
File: smartcash/ui/dataset/preprocessing/handlers/persistence_handler.py
Deskripsi: Handler untuk persistensi konfigurasi preprocessing
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger("preprocessing_persistence")

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
        # Import fungsi get_config_manager
        from smartcash.common.config.manager import get_config_manager
        
        # Dapatkan instance ConfigManager menggunakan fungsi singleton
        config_manager = get_config_manager()
        
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
    # Import parameter extractor
    from smartcash.ui.dataset.preprocessing.handlers.parameter_handler import extract_preprocess_params
    
    # Ekstrak parameter preprocessing dari UI
    preprocess_params = extract_preprocess_params(ui_components)
    
    # Dapatkan split dari UI
    split_option = ui_components.get('split_selector', {}).value if 'split_selector' in ui_components else 'All Splits'
    split_map = {'All Splits': None, 'Train Only': 'train', 'Validation Only': 'valid', 'Test Only': 'test'}
    split = split_map.get(split_option)
    
    # Tambahkan split ke parameter
    preprocess_params['split'] = split
    
    return preprocess_params

def sync_config_with_drive(ui_components: Dict[str, Any]) -> bool:
    """
    Sinkronisasi konfigurasi dengan file di drive.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Import fungsi get_config_manager
        from smartcash.common.config.manager import get_config_manager
        
        # Dapatkan instance ConfigManager menggunakan fungsi singleton
        config_manager = get_config_manager()
        
        # Dapatkan konfigurasi dari UI
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import update_config_from_ui
        updated_config = update_config_from_ui(ui_components, ui_components.get('config', {}))
        
        # Simpan konfigurasi ke drive
        success = config_manager.save_module_config('preprocessing', updated_config)
        
        # Log info
        if success:
            logger.info(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan drive")
        else:
            logger.error(f"{ICONS['error']} Gagal menyinkronkan konfigurasi dengan drive")
        
        return success
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat menyinkronkan konfigurasi: {str(e)}")
        return False

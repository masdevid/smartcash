"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Handler terpadu untuk preprocessing dataset menggunakan shared modules
"""

from typing import Dict, Any
import os
from pathlib import Path
from IPython.display import display, clear_output

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup semua handler untuk komponen UI preprocessing dataset dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import komponen standard
    from smartcash.ui.utils.constants import ICONS

    # Dapatkan logger
    logger = ui_components.get('logger')
    
    try:
        # Persiapan awal - sertakan nilai awal di ui_components
        preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        data_dir = config.get('data', {}).get('dir', 'data')
        
        # Update ui_components dengan path standar yang diperlukan
        ui_components.update({
            'data_dir': data_dir,
            'preprocessed_dir': preprocessed_dir,
        })
        
        # Setup config handler (tetap spesifik untuk preprocessing)
        from smartcash.ui.dataset.preprocessing_config_handler import setup_preprocessing_config_handler
        ui_components = setup_preprocessing_config_handler(ui_components, config, env)
        
        # Setup tombol handler (tetap spesifik untuk preprocessing)
        from smartcash.ui.dataset.preprocessing_click_handler import setup_click_handlers
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Setup dataset manager
        from smartcash.ui.dataset.shared.setup_utils import setup_manager
        dataset_manager = setup_manager(ui_components, config, 'preprocessing')
        ui_components['dataset_manager'] = dataset_manager
        
        # Gunakan shared handlers untuk progress, visualization, cleanup dan summary
        from smartcash.ui.dataset.shared.integration import apply_shared_handlers, create_cleanup_function
        ui_components = apply_shared_handlers(ui_components, env, config, 'preprocessing')
        create_cleanup_function(ui_components, 'preprocessing')
        
        # Cek status data preprocessed yang sudah ada
        from smartcash.ui.dataset.shared.setup_utils import detect_module_state
        ui_components = detect_module_state(ui_components, 'preprocessing')
        
        # Log success
        if logger: logger.info(f"{ICONS['success']} Handler preprocessing berhasil diinisialisasi")
        
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        if 'status' in ui_components:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error setup handler: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error setup preprocessing handler: {str(e)}")
    
    return ui_components

def get_preprocessing_dataset_manager(config: Dict[str, Any], logger=None):
    """
    Dapatkan dataset manager untuk preprocessing dengan fallback yang terstandarisasi.
    
    Args:
        config: Konfigurasi aplikasi
        logger: Logger
        
    Returns:
        Dataset manager instance atau None
    """
    # Gunakan fallback_utils untuk konsistensi
    from smartcash.ui.utils.fallback_utils import get_dataset_manager
    return get_dataset_manager(config, logger)
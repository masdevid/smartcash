"""
File: smartcash/ui/dataset/split_config_handler.py
Deskripsi: Handler utama untuk konfigurasi split dataset yang menggunakan komponen modular
"""

from typing import Dict, Any
import os

def setup_split_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan logger jika tersedia
    logger = None
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger("split_config")
    except ImportError:
        logger = None
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    try:
        # Import komponen-komponen modular
        from smartcash.ui.dataset.split_config_utils import load_dataset_config
        from smartcash.ui.dataset.split_config_handlers import register_handlers, initialize_ui
        
        # Load dataset config dan gabungkan dengan config utama
        dataset_config = load_dataset_config('configs/dataset_config.yaml')
        
        # Gabungkan dengan config utama
        if 'data' in dataset_config:
            config['data'].update(dataset_config.get('data', {}))
            
            if logger:
                logger.info(f"🔄 Dataset config berhasil dimuat dari configs/dataset_config.yaml")
        
        # Register handlers untuk berbagai komponen UI
        ui_components = register_handlers(ui_components, config, env, logger)
        
        # Initialize UI dengan data dari config
        initialize_ui(ui_components, config, env, logger)
        
    except Exception as e:
        # Fallback to minimal setup in case of errors
        if 'output_box' in ui_components:
            with ui_components['output_box']:
                from smartcash.ui.components.alerts import create_status_indicator
                from IPython.display import display
                display(create_status_indicator("error", f"Error saat setup split config: {str(e)}"))
                
        if logger:
            logger.error(f"❌ Error saat setup split config: {str(e)}")
    
    return ui_components
"""
File: smartcash/ui/dataset/split_config_handler.py
Deskripsi: Handler utama untuk konfigurasi split dataset yang menggunakan komponen modular
"""

from typing import Dict, Any
import logging

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
    # Setup logging terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "cell_split_config", log_level=logging.INFO)
        if logger:
            ui_components['logger'] = logger
            logger.info(f"🚀 Split config handler diinisialisasi")
    except Exception as e:
        print(f"⚠️ Tidak dapat setup logging: {str(e)}")
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    try:
        # Import handler dan inisialisasi UI
        from smartcash.ui.dataset.split_config_handlers import register_handlers
        from smartcash.ui.dataset.split_config_initialization import initialize_ui
        
        # Register handlers untuk berbagai komponen UI
        ui_components = register_handlers(ui_components, config, env, logger)
        
        # Initialize UI dengan data dari config
        initialize_ui(ui_components, config, env, logger)
        
    except Exception as e:
        # Tampilkan pesan error sederhana
        if 'output_box' in ui_components:
            with ui_components['output_box']:
                from smartcash.ui.components.alerts import create_status_indicator
                from smartcash.ui.utils.constants import ICONS
                from IPython.display import display
                display(create_status_indicator("error", f"{ICONS['error']} Error saat setup split config: {str(e)}"))
                
        if logger:
            logger.error(f"❌ Error saat setup split config: {str(e)}")
    
    return ui_components
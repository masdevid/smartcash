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
        logger = setup_ipython_logging(ui_components, "split_config", log_level=logging.INFO)
        if logger:
            ui_components['logger'] = logger
            logger.info(f"🚀 Split config handler diinisialisasi")
    except ImportError as e:
        print(f"⚠️ Tidak dapat setup logging: {str(e)}")
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    try:
        # Register handlers untuk berbagai komponen UI
        from smartcash.ui.dataset.split_config_handlers import register_handlers
        ui_components = register_handlers(ui_components, config, env, logger)
        
        # Initialize UI dengan data dari config
        from smartcash.ui.dataset.split_config_initialization import initialize_ui
        initialize_ui(ui_components, config, env, logger)
        
        # Detect dan sync Google Drive jika perlu
        if 'drive_options' in ui_components and ui_components['drive_options'].children[0].value:
            if logger: logger.info(f"🔄 Google Drive aktif, mendeteksi dataset...")
            
            # Cek apakah perlu sync otomatis
            sync_on_change = config.get('data', {}).get('sync_on_change', True)
            if sync_on_change:
                if logger: logger.info(f"🔄 Sinkronisasi drive otomatis diaktifkan")
                
                def sync_callback(status, message):
                    from smartcash.ui.components.alerts import create_status_indicator
                    with ui_components['output_box']:
                        display(create_status_indicator(status, message))
                
                # Import dan jalankan sinkronisasi async
                from smartcash.ui.utils.drive_detector import async_sync_drive
                async_sync_drive(config, env, logger, sync_callback)
        
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
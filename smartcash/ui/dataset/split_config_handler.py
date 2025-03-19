"""
File: smartcash/ui/dataset/split_config_handler.py
Deskripsi: Handler utama untuk konfigurasi split dataset dengan validasi komponen yang lebih baik
"""

from typing import Dict, Any, Optional
import logging
from IPython.display import display, HTML

def setup_split_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI konfigurasi split dataset dengan validasi.
    
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
    
    # Validasi komponen UI yang diperlukan
    required_components = ['split_sliders', 'stratified', 'save_button', 'reset_button', 'output_box']
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        error_msg = f"Komponen UI tidak lengkap. Komponen yang hilang: {', '.join(missing_components)}"
        if logger:
            logger.error(f"❌ {error_msg}")
        
        # Display error dalam output box jika tersedia
        if 'output_box' in ui_components:
            from smartcash.ui.utils.constants import ICONS
            with ui_components['output_box']:
                display(HTML(f"""
                <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px;">
                    <p>{ICONS.get('error', '❌')} {error_msg}</p>
                </div>
                """))
        return ui_components
    
    try:
        # Register handlers untuk berbagai komponen UI
        from smartcash.ui.dataset.split_config_handlers import register_handlers
        ui_components = register_handlers(ui_components, config, env, logger)
        
        # Initialize UI dengan data dari config
        from smartcash.ui.dataset.split_config_initialization import initialize_ui
        ui_components = initialize_ui(ui_components, config, env, logger)
        
        # Detect dan sync Google Drive jika perlu
        if 'drive_options' in ui_components and hasattr(ui_components['drive_options'], 'children') and len(ui_components['drive_options'].children) > 0:
            use_drive = ui_components['drive_options'].children[0].value
            if use_drive:
                if logger: logger.info(f"🔄 Google Drive aktif, mendeteksi dataset...")
                
                # Cek apakah perlu sync otomatis
                sync_on_change = config.get('data', {}).get('sync_on_change', True)
                if sync_on_change:
                    if logger: logger.info(f"🔄 Sinkronisasi drive otomatis diaktifkan")
                    
                    def sync_callback(status, message):
                        if 'output_box' in ui_components:
                            with ui_components['output_box']:
                                from smartcash.ui.components.alerts import create_status_indicator
                                display(create_status_indicator(status, message))
                    
                    # Import dan jalankan sinkronisasi async
                    try:
                        from smartcash.ui.utils.drive_detector import async_sync_drive
                        async_sync_drive(config, env, logger, sync_callback)
                    except ImportError as e:
                        if logger: logger.warning(f"⚠️ Tidak dapat sinkronisasi drive: {str(e)}")
        
    except Exception as e:
        # Tampilkan pesan error sederhana
        if 'output_box' in ui_components:
            with ui_components['output_box']:
                from smartcash.ui.utils.constants import ICONS
                display(HTML(f"""
                <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px;">
                    <p>{ICONS.get('error', '❌')} Error saat setup split config: {str(e)}</p>
                </div>
                """))
                
        if logger:
            logger.error(f"❌ Error saat setup split config: {str(e)}")
    
    return ui_components
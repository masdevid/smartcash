"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk modul konfigurasi environment dengan pendekatan modular dan efisien
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

def initialize_env_config_ui():
    """
    Inisialisasi UI dan handler untuk konfigurasi environment dengan pendekatan cell template.
    
    Returns:
        Dictionary komponen UI yang terinisialisasi
    """
    try:
        # Gunakan cell template standar untuk konsistensi
        from smartcash.ui.cell_template import setup_cell
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
        
        # Inisialisasi async opsional untuk verifikasi konfigurasi
        def init_async(ui_components, env, config):
            try:
                # Verifikasi konfigurasi default
                from smartcash.ui.setup.drive_sync_initializer import initialize_configs
                logger = ui_components.get('logger')
                
                # Jalankan sinkronisasi konfigurasi
                success, message = initialize_configs(logger)
                if logger:
                    logger.info(f"🔄 Sinkronisasi konfigurasi: {message}")
            except Exception as e:
                logger = ui_components.get('logger')
                if logger:
                    logger.warning(f"⚠️ Error saat inisialisasi konfigurasi: {str(e)}")
        
        # Gunakan cell template untuk setup standar
        ui_components = setup_cell(
            cell_name="env_config",
            create_ui_func=create_env_config_ui,
            setup_handlers_func=setup_env_config_handlers,
            init_async_func=init_async
        )
        
        return ui_components
        
    except Exception as e:
        # Fallback jika cell template gagal
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = {'module_name': 'env_config'}
        ui_components = create_fallback_ui(
            ui_components, 
            f"❌ Error inisialisasi environment config: {str(e)}", 
            "error"
        )
        return ui_components
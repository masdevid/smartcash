"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan integrasi sinkronisasi Drive
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi utilities"""
    ui_components = {}
    logger = None
    
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, reset_logging

        cell_name = "env_config"
        # Setup environment dan load config dengan utils terstandarisasi
        env, config = setup_notebook_environment(cell_name)
        
        # Setup logging terlebih dahulu sebelum melakukan apapun
        ui_components = {'status': widgets.Output(), 'module_name': cell_name}
        logger = setup_ipython_logging(ui_components, cell_name)
        ui_components['logger'] = logger
        
        # Log awal untuk konfirmasi bahwa logging berfungsi
        if logger: 
            logger.info("🚀 Inisialisasi environment config")
        
        # Pastikan konfigurasi default tersedia dengan utils terstandarisasi
        try:
            from smartcash.common.default_config import ensure_all_configs_exist
            success = ensure_all_configs_exist()
            if logger:
                if success:
                    logger.info("✅ Konfigurasi default berhasil diverifikasi")
                else:
                    logger.info("ℹ️ Konfigurasi default sudah ada")
        except ImportError as e:
            if logger: logger.warning(f"⚠️ Module default_config tidak tersedia: {str(e)}")
        
        # Buat komponen UI dengan utils terstandarisasi
        ui_components = create_env_config_ui(env, config)
        
        # Pastikan logger tetap ada di ui_components
        if logger and 'logger' not in ui_components:
            ui_components['logger'] = logger
        elif not logger and 'logger' in ui_components:
            logger = ui_components['logger']
        
        # Log untuk visualisasi progress setup
        if logger: logger.info("🏗️ Komponen UI environment config berhasil dibuat")
        
        # Inisialisasi Drive sync menggunakan modul terpisah
        from smartcash.ui.setup.drive_sync_initializer import initialize_drive_sync
        initialize_drive_sync(ui_components)
        if logger: logger.info("🔄 Drive sync berhasil diinisialisasi")

        # Setup handlers untuk UI
        try:
            ui_components = setup_env_config_handlers(ui_components, env, config)
            if logger: logger.info("🚀 Handlers environment config berhasil dimuat")
        except Exception as e:
            if logger: logger.error(f"❌ Gagal memuat handlers: {str(e)}")
        
        # Cleanup function untuk dijalankan saat cell di-reset
        def cleanup_resources():
            """Clean up resources properly."""
            # Unregister observer group jika ada
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            
            # Kembalikan logging ke default
            reset_logging()
            
            if logger: logger.debug("🧹 Environment config resources cleaned up")
            
        # Register cleanup function
        ui_components['cleanup'] = cleanup_resources
        
        # Register cleanup untuk event IPython jika di notebook
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            get_ipython().events.register('pre_run_cell', cleanup)

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    except Exception as e:
        # Generic exception handling
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"❌ Error saat inisialisasi environment config: {str(e)}", "error", ui_components)
        
        # Pastikan logging dikembalikan ke default
        try:
            reset_logging()
        except:
            pass
    
    return ui_components
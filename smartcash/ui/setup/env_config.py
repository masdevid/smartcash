"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan perbaikan integrasi logging
"""

from typing import Dict, Any
from IPython.display import display, HTML

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi logging yang ditingkatkan"""
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, log_to_ui

        # Setup environment dan load config
        cell_name = "env_config"
        env, config = setup_notebook_environment(cell_name)
        
        # Buat komponen UI dengan utils terstandarisasi
        ui_components = create_env_config_ui(env, config)
        
        # Kirim pesan log awal untuk konfirmasi visibilitas
        if 'status' in ui_components:
            log_to_ui(ui_components, "🚀 Inisialisasi environment config dimulai", "info")
        
        # Setup logging dengan integrasi UI
        logger = setup_ipython_logging(ui_components, cell_name)
        if logger:
            ui_components['logger'] = logger
            logger.info("✅ Logger environment config berhasil diinisialisasi")
        
        # Pastikan konfigurasi default tersedia
        try:
            from smartcash.common.default_config import ensure_all_configs_exist
            success = ensure_all_configs_exist()
            if logger:
                if success: logger.info("✅ Konfigurasi default berhasil diverifikasi")
                else: logger.info("ℹ️ Konfigurasi default sudah ada")
        except ImportError as e:
            if logger: logger.warning(f"⚠️ Module default_config tidak tersedia: {str(e)}")
        
        # Inisialisasi Drive sync menggunakan modul terpisah
        try:
            from smartcash.ui.setup.drive_sync_initializer import initialize_drive_sync
            initialize_drive_sync(ui_components)
            if logger: logger.info("🔄 Drive sync berhasil diinisialisasi")
        except ImportError as e:
            if logger: logger.warning(f"⚠️ Module drive_sync_initializer tidak tersedia: {str(e)}")

        # Setup handlers untuk UI
        ui_components = setup_env_config_handlers(ui_components, env, config)
        if logger: logger.info("🚀 Handlers environment config berhasil disetup")
        
        # Cleanup function untuk dijalankan saat cell di-reset
        def cleanup_resources():
            """Clean up resources properly."""
            # Unregister observer group jika ada
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception as e:
                    if logger: logger.debug(f"⚠️ Error saat unregister observer: {str(e)}")
            
            # Kembalikan logging ke default
            from smartcash.ui.utils.logging_utils import reset_logging
            reset_logging()
            
            if logger: logger.debug("🧹 Environment config resources cleaned up")
            
        # Register cleanup function
        ui_components['cleanup'] = cleanup_resources
        
        # Register cleanup untuk event IPython jika di notebook
        try:
            from IPython import get_ipython
            if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
                cleanup = ui_components['cleanup']
                get_ipython().events.register('pre_run_cell', cleanup)
                if logger: logger.debug("✅ Cleanup event berhasil diregistrasi")
        except Exception as e:
            if logger: logger.debug(f"⚠️ Error saat register cleanup event: {str(e)}")

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        ui_components = {'status': None, 'module_name': 'env_config'}
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    except Exception as e:
        # Generic exception handling
        from smartcash.ui.utils.fallback_utils import show_status
        ui_components = {'status': None, 'module_name': 'env_config'}
        show_status(f"❌ Error saat inisialisasi environment config: {str(e)}", "error", ui_components)
        
        # Pastikan logging dikembalikan ke default
        try:
            from smartcash.ui.utils.logging_utils import reset_logging
            reset_logging()
        except:
            pass
    
    return ui_components
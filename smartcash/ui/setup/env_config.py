"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan inisialisasi logging yang ditingkatkan
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan inisialisasi logging yang ditingkatkan"""
    # Inisialisasi ui_components lebih awal untuk dukungan logging
    ui_components = {
        'status': widgets.Output(
            layout=widgets.Layout(
                width='100%',
                border='1px solid #ddd',
                min_height='100px',
                max_height='300px',
                margin='10px 0',
                padding='10px',
                overflow='auto'
            )
        ),
        'module_name': 'env_config'
    }
    
    # Keluarkan log direct ke output widget untuk memastikan visibilitas
    with ui_components['status']:
        display(HTML("<div style='color: blue;'><strong>🚀 Inisialisasi environment config (via direct HTML)...</strong></div>"))
    
    # Setup logger lebih awal
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, reset_logging, log_to_ui
        
        # Log langsung ke UI sebelum logger disetup
        log_to_ui(ui_components, "🔄 Mengatur logging system...", "info")
        
        # Setup logger dengan UI components
        logger = setup_ipython_logging(ui_components, 'env_config')
        ui_components['logger'] = logger
        
        # Tes logger untuk verifikasi
        if logger:
            logger.info("🚀 Inisialisasi environment config via logger")
            logger.debug("🔍 Debug log test")
            logger.success("✅ Logger setup berhasil")
        
    except ImportError as e:
        with ui_components['status']:
            display(HTML(f"<div style='color: red;'><strong>⚠️ Error saat setup logging: {str(e)}</strong></div>"))
        print(f"⚠️ Error saat setup logging: {str(e)}")
    
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers

        # Setup environment dan load config
        cell_name = "env_config"
        env, config = setup_notebook_environment(cell_name)
        
        if logger: logger.debug(f"🔍 Environment dan konfigurasi berhasil dimuat")
        
        # Pastikan konfigurasi default tersedia
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
        new_ui_components = create_env_config_ui(env, config)
        
        # Gabungkan komponen lama (dengan logger) dan komponen baru
        for key, value in new_ui_components.items():
            ui_components[key] = value
        
        # Pastikan status widget tetap sama untuk konsistensi logging
        if 'status' in new_ui_components and logger:
            # Salin log lama ke status baru jika berbeda
            if id(new_ui_components['status']) != id(ui_components['status']):
                with ui_components['status']:
                    old_content = ui_components['status']._repr_mimebundle_()
                
                # Reset status widget dan tampilkan konten lama
                with new_ui_components['status']:
                    if old_content:
                        try:
                            display(HTML(old_content.get('text/html', '')))
                        except:
                            pass
        
        # Pastikan logger tetap ada di ui_components
        if logger:
            ui_components['logger'] = logger
            logger.info("🏗️ Komponen UI environment config berhasil dibuat")
        
        # Inisialisasi Drive sync menggunakan modul terpisah
        try:
            from smartcash.ui.setup.drive_sync_initializer import initialize_drive_sync
            initialize_drive_sync(ui_components)
            if logger: logger.info("🔄 Drive sync berhasil diinisialisasi")
        except ImportError as e:
            if logger: logger.warning(f"⚠️ Module drive_sync_initializer tidak tersedia: {str(e)}")

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
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception as e:
                    if logger: logger.debug(f"⚠️ Error saat unregister observer: {str(e)}")
            
            # Kembalikan logging ke default
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
        except Exception as e:
            if logger: logger.debug(f"⚠️ Error saat register cleanup event: {str(e)}")

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
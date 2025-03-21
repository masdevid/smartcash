"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan integrasi sinkronisasi Drive dan pembuatan config otomatis
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, Any, Optional

def initialize_drive_sync(ui_components=None):
    """
    Inisialisasi dan sinkronisasi Google Drive dengan progress tracking
    
    Args:
        ui_components: Dictionary komponen UI yang akan diupdate (opsional)
    """
    # Pastikan output tersedia
    if not ui_components or 'status' not in ui_components:
        print("Error: Output widget tidak tersedia")
        return
    
    # Gunakan output widget untuk semua log
    status_output = ui_components['status']
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    # Fungsi log yang mengarahkan ke output widget
    def log(message, status_type="info"):
        with status_output:
            from smartcash.ui.utils.alert_utils import create_info_alert
            display(create_info_alert(message, status_type))
    
    # Fungsi update progress
    def update_progress(step, message, status_type="info", total_steps=4):
        log(message, status_type)
        
        # Update progress bar jika tersedia
        if progress_bar:
            progress_bar.value = step
            progress_bar.max = total_steps
        
        # Update progress message jika tersedia
        if progress_message:
            progress_message.value = message
    
    try:
        import os, sys
        from smartcash.ui.utils.constants import ICONS
        
        # Step 1: Deteksi environment
        update_progress(1, "Mendeteksi environment Google Drive...")
        is_colab = 'google.colab' in sys.modules
        drive_mounted = os.path.exists('/content/drive/MyDrive')
        
        if not is_colab:
            update_progress(4, "Bukan lingkungan Google Colab, lewati sinkronisasi Drive", "info")
            return
        
        # Step 2: Pastikan config default ada
        update_progress(2, "Memastikan file konfigurasi default tersedia...", "info")
        try:
            # Import modul untuk membuat config default
            from smartcash.common.default_config import ensure_all_configs_exist
            configs_created = ensure_all_configs_exist()
            
            if configs_created:
                log("File konfigurasi default berhasil dibuat", "success")
            else:
                log("File konfigurasi default sudah tersedia", "info")
        except Exception as e:
            log(f"Gagal membuat konfigurasi default: {str(e)}", "warning")
        
        # Step 3: Mount Drive jika perlu
        if not drive_mounted:
            update_progress(3, "Mounting Google Drive...", "info")
            
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                drive_mounted = os.path.exists('/content/drive/MyDrive')
                
                if not drive_mounted:
                    update_progress(4, "Gagal mounting Google Drive", "error")
                    return
                
                log("Google Drive berhasil dimount", "success")
            except Exception as e:
                update_progress(4, f"Error saat mounting Google Drive: {str(e)}", "error")
                return
        else:
            log("Google Drive sudah terhubung", "success")
        
        # Step 4: Sinkronisasi konfigurasi
        update_progress(4, "Sinkronisasi konfigurasi dengan Drive...", "info")
        
        try:
            from smartcash.common.config_sync import sync_all_configs
            results = sync_all_configs(sync_strategy='drive_priority')
            
            success_count = len(results.get("success", []))
            failure_count = len(results.get("failure", []))
            
            if failure_count == 0:
                log(f"Sinkronisasi berhasil: {success_count} file ✓", "success")
            else:
                log(f"Sinkronisasi: {success_count} berhasil, {failure_count} gagal", "warning")
        except ImportError:
            log("Modul config_sync tidak tersedia, lewati sinkronisasi", "warning")
        
    except Exception as e:
        log(f"Error: {str(e)}", "error")

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi utilities"""
    
    # Import komponen dengan pendekatan konsolidasi
    from smartcash.ui.setup.env_config_component import create_env_config_ui
    from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component
    from smartcash.ui.utils.logging_utils import setup_ipython_logging
    
    try:
        # Setup notebook environment
        env, config = setup_notebook_environment("env_config")
        
        # Pastikan konfigurasi default tersedia
        from smartcash.common.default_config import ensure_all_configs_exist
        ensure_all_configs_exist()
        
        # Buat komponen UI dengan helpers
        ui_components = create_env_config_ui(env, config)
        
        # Tambahkan progress tracker jika belum ada
        if 'progress_bar' not in ui_components:
            ui_components['progress_bar'] = widgets.IntProgress(
                value=0,
                min=0,
                max=4,
                description='Progress:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='50%', margin='10px 0')
            )
            ui_components['progress_message'] = widgets.HTML("Mempersiapkan environment...")
            
            # Tambahkan ke UI
            if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
                # Cari posisi setelah header
                children = list(ui_components['ui'].children)
                header_pos = 0
                for i, child in enumerate(children):
                    if child == ui_components.get('header'):
                        header_pos = i
                        break
                
                # Tambahkan progress components setelah header
                tracker_box = widgets.VBox([
                    ui_components['progress_bar'],
                    ui_components['progress_message']
                ])
                children.insert(header_pos + 1, tracker_box)
                ui_components['ui'].children = children
        
        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, "env_config")
        
        # Fungsi log ke output widget
        def log_to_output(message, status_type="info"):
            if 'status' in ui_components:
                with ui_components['status']:
                    from smartcash.ui.utils.alert_utils import create_info_alert
                    display(create_info_alert(message, status_type))
        
        # Inisialisasi drive dengan komponen UI yang baru dibuat
        initialize_drive_sync(ui_components)
        
        # Setup handlers untuk UI
        try:
            ui_components = setup_env_config_handlers(ui_components, env, config)
            log_to_output("🚀 Handlers environment config berhasil dimuat", "success")
        except Exception as e:
            log_to_output(f"❌ Gagal memuat handlers: {str(e)}", "error")
        
        # Cek fungsionalitas drive_handler
        try:
            from smartcash.ui.setup.drive_handler import setup_drive_handler
            ui_components = setup_drive_handler(ui_components, env, config, auto_connect=True)
            log_to_output("🔗 Drive handler berhasil diinisialisasi", "success")
        except ImportError as e:
            log_to_output(f"⚠️ Module drive_handler tidak tersedia: {str(e)}", "warning")
        
    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import import_with_fallback, show_status
        
        # Pastikan konfigurasi default tersedia
        from smartcash.common.default_config import ensure_all_configs_exist
        ensure_all_configs_exist()
        
        # Fallback environment setup
        env = type('DummyEnv', (), {
            'is_colab': 'google.colab' in __import__('sys').modules,
            'base_dir': __import__('os').getcwd(),
            'is_drive_mounted': False,
        })
        config = {}
        
        # Buat UI components
        ui_components = create_env_config_ui(env, config)
        
        # Tambahkan progress tracker
        ui_components['progress_bar'] = widgets.IntProgress(
            value=0, min=0, max=4, description='Progress:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%', margin='10px 0')
        )
        ui_components['progress_message'] = widgets.HTML("Mempersiapkan environment...")
        
        # Coba inisialisasi drive dengan komponen UI fallback
        try:
            initialize_drive_sync(ui_components)
        except Exception:
            pass
        
        # Tampilkan pesan error
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components
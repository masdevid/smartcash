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
    status_output = None
    progress_bar = None
    progress_message = None
    
    # Cek komponen UI yang tersedia
    if ui_components:
        if 'status' in ui_components:
            status_output = ui_components['status']
        if 'progress_bar' in ui_components:
            progress_bar = ui_components['progress_bar']
        if 'progress_message' in ui_components:
            progress_message = ui_components['progress_message']
    
    # Inisialisasi progress tracking
    total_steps = 4  # Detect, Mount, Create Default Config, Sync
    current_step = 0
    
    def update_progress(step, message, status_type="info"):
        nonlocal current_step
        current_step = step
        
        # Update progress bar
        if progress_bar:
            progress_bar.value = step
            progress_bar.max = total_steps
            
        # Update message
        if progress_message:
            progress_message.value = message
            
        # Tampilkan di output
        if status_output:
            with status_output:
                from smartcash.ui.utils.alert_utils import create_info_alert
                display(create_info_alert(message, status_type))
    
    try:
        import os, sys
        from smartcash.ui.utils.constants import ICONS
        
        # Step 1: Deteksi environment
        update_progress(1, "Mendeteksi environment Google Drive...")
        is_colab = 'google.colab' in sys.modules
        drive_mounted = os.path.exists('/content/drive/MyDrive')
        
        if not is_colab:
            update_progress(total_steps, "Bukan lingkungan Google Colab, lewati sinkronisasi Drive", "info")
            return
        
        # Step 2: Pastikan config default ada
        update_progress(2, "Memastikan file konfigurasi default tersedia...", "info")
        try:
            # Import modul untuk membuat config default
            from smartcash.common.default_config import ensure_all_configs_exist
            configs_created = ensure_all_configs_exist()
            
            if configs_created:
                update_progress(2, "File konfigurasi default berhasil dibuat", "success")
            else:
                update_progress(2, "File konfigurasi default sudah tersedia", "info")
        except Exception as e:
            update_progress(2, f"Gagal membuat konfigurasi default: {str(e)}", "warning")
        
        # Step 3: Mount Drive jika perlu
        if not drive_mounted:
            update_progress(3, "Mounting Google Drive...", "info")
            
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                drive_mounted = os.path.exists('/content/drive/MyDrive')
                
                if not drive_mounted:
                    update_progress(3, "Gagal mounting Google Drive", "error")
                    return
                
                update_progress(3, "Google Drive berhasil dimount", "success")
            except Exception as e:
                update_progress(3, f"Error saat mounting Google Drive: {str(e)}", "error")
                return
        else:
            update_progress(3, "Google Drive sudah terhubung", "success")
        
        # Step 4: Sinkronisasi konfigurasi
        update_progress(4, "Sinkronisasi konfigurasi dengan Drive...", "info")
        
        try:
            from smartcash.common.config_sync import sync_all_configs
            results = sync_all_configs(sync_strategy='drive_priority')
            
            success_count = len(results.get("success", []))
            failure_count = len(results.get("failure", []))
            
            if failure_count == 0:
                update_progress(total_steps, f"Sinkronisasi berhasil: {success_count} file ✓", "success")
            else:
                update_progress(total_steps, f"Sinkronisasi: {success_count} berhasil, {failure_count} gagal", "warning")
        except ImportError:
            update_progress(total_steps, "Modul config_sync tidak tersedia, lewati sinkronisasi", "warning")
        
    except Exception as e:
        if status_output:
            with status_output:
                from smartcash.ui.utils.alert_utils import create_info_alert
                display(create_info_alert(f"Error: {str(e)}", "error"))

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
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Modul environment config berhasil dimuat")
        
        # Inisialisasi drive dengan komponen UI yang baru dibuat
        initialize_drive_sync(ui_components)
        
        # Setup handlers untuk UI
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
        # Cek fungsionalitas drive_handler
        try:
            from smartcash.ui.setup.drive_handler import setup_drive_handler
            ui_components = setup_drive_handler(ui_components, env, config, auto_connect=True)
        except ImportError as e:
            if logger:
                logger.debug(f"Module drive_handler tidak tersedia: {str(e)}")
        
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
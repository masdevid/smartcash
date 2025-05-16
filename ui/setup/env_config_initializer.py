"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk konfigurasi environment
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional
from IPython.display import display

from smartcash.ui.setup.env_config_component import create_env_config_ui
from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.ui_logger import create_direct_ui_logger
from smartcash.ui.setup.ui_helpers import disable_ui_during_processing, cleanup_ui

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi environment
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Inisialisasi environment manager
    env_manager = get_environment_manager()
    
    # Inisialisasi config manager
    config_manager = get_config_manager()
    
    # Buat komponen UI
    ui_components = create_env_config_ui(env_manager, config_manager)
    
    # Setup logger
    logger = create_direct_ui_logger(ui_components, "env_config")
    ui_components['logger'] = logger
    
    # Setup handlers
    setup_env_config_handlers(ui_components, env_manager, config_manager)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    # Log informasi
    logger.info("Environment config handlers berhasil diinisialisasi")
    
    # Otomatisasi pemeriksaan environment dan sinkronisasi konfigurasi
    import threading
    import time
    from smartcash.ui.utils.alert_utils import create_success_box
    
    def auto_check_and_sync():
        # Tunggu sebentar agar UI logger siap
        time.sleep(1)
        
        try:
            # Update status panel untuk menunjukkan proses sedang berjalan
            ui_components['status_panel'].value = create_success_box(
                "Pemeriksaan Environment", 
                "Sedang memeriksa environment dan melakukan sinkronisasi konfigurasi..."
            ).value
            
            # Tampilkan progress bar
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_message'].layout.visibility = 'visible'
            ui_components['progress_message'].value = "Memeriksa environment..."
            ui_components['progress_bar'].value = 3
            
            # Periksa environment tanpa log berlebihan
            env_info = env_manager.get_system_info()
            ui_components['progress_bar'].value = 5
            ui_components['progress_message'].value = "Melakukan sinkronisasi konfigurasi..."
            
            # Sinkronisasi konfigurasi tanpa log
            if hasattr(env_manager, 'sync_config'):
                env_manager.sync_config()
            ui_components['progress_bar'].value = 8
            
            # Simpan konfigurasi tanpa log
            if hasattr(env_manager, 'save_environment_config'):
                env_manager.save_environment_config()
            
            # Selesai
            ui_components['progress_bar'].value = 10
            ui_components['progress_message'].value = "Pemeriksaan dan sinkronisasi selesai"
            
            # Update status panel dengan hasil
            ui_components['status_panel'].value = create_success_box(
                "Konfigurasi Environment", 
                "Pemeriksaan environment dan sinkronisasi konfigurasi berhasil dilakukan."
            ).value
            
            # Sembunyikan progress bar setelah beberapa detik
            time.sleep(2)
            ui_components['reset_progress']()
            
            # Log ringkasan
            logger.info("✅ Pemeriksaan environment dan sinkronisasi konfigurasi berhasil")
        except Exception as e:
            # Update status panel dengan error
            from smartcash.ui.utils.alert_utils import create_error_box
            ui_components['status_panel'].value = create_error_box(
                "Error", 
                f"Terjadi kesalahan saat pemeriksaan environment: {str(e)}"
            ).value
            logger.error(f"❌ Error saat otomatisasi: {str(e)}")
            ui_components['reset_progress']()
    
    # Jalankan pemeriksaan dan sinkronisasi di thread terpisah
    auto_thread = threading.Thread(target=auto_check_and_sync)
    auto_thread.daemon = True
    auto_thread.start()
    
    return ui_components

# Fungsi _disable_ui_during_processing dan _cleanup_ui dipindahkan ke ui_helpers.py

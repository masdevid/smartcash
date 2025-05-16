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
    from smartcash.ui.utils.alert_utils import create_info_box
    
    def auto_check_and_sync():
        # Tunggu sebentar agar UI logger siap
        time.sleep(1)
        
        try:
            # Update status panel untuk menunjukkan proses sedang berjalan
            ui_components['status_panel'].value = create_info_box(
                "Pemeriksaan Environment", 
                "Sedang memeriksa environment dan melakukan sinkronisasi konfigurasi...",
                style="info"
            ).value
            
            # Periksa environment tanpa log berlebihan
            env_info = env_manager.get_system_info()
            
            # Sinkronisasi konfigurasi tanpa log
            if hasattr(env_manager, 'sync_config'):
                env_manager.sync_config()
            
            # Simpan konfigurasi tanpa log
            if hasattr(env_manager, 'save_environment_config'):
                env_manager.save_environment_config()
            
            # Update status panel dengan hasil
            ui_components['status_panel'].value = create_info_box(
                "Konfigurasi Environment", 
                "Pemeriksaan environment dan sinkronisasi konfigurasi berhasil dilakukan.",
                style="success"
            ).value
            
            # Tunggu sebentar sebelum menampilkan pesan sukses
            time.sleep(1)
            
            # Log ringkasan
            logger.info("✅ Pemeriksaan environment dan sinkronisasi konfigurasi berhasil")
        except Exception as e:
            # Update status panel dengan error
            ui_components['status_panel'].value = create_info_box(
                "Error", 
                f"Terjadi kesalahan saat pemeriksaan environment: {str(e)}",
                style="error"
            ).value
            logger.error(f"❌ Error saat otomatisasi: {str(e)}")
    
    # Jalankan pemeriksaan dan sinkronisasi di thread terpisah
    auto_thread = threading.Thread(target=auto_check_and_sync)
    auto_thread.daemon = True
    auto_thread.start()
    
    return ui_components

# Fungsi _disable_ui_during_processing dan _cleanup_ui dipindahkan ke ui_helpers.py

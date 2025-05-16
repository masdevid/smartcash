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
    
    def auto_check_and_sync():
        # Tunggu sebentar agar UI logger siap
        time.sleep(2)
        
        try:
            # Periksa environment
            logger.info("Memeriksa environment secara otomatis...")
            env_info = env_manager.check_environment()
            logger.info("Environment berhasil diperiksa")
            
            # Sinkronisasi konfigurasi
            logger.info("Menyinkronkan konfigurasi secara otomatis...")
            if hasattr(env_manager, 'sync_config'):
                success, message = env_manager.sync_config()
                if success:
                    logger.info(message)
                else:
                    logger.warning(message)
            
            # Simpan konfigurasi
            logger.info("Menyimpan konfigurasi secara otomatis...")
            if hasattr(env_manager, 'save_environment_config'):
                success, message = env_manager.save_environment_config()
                if success:
                    logger.info(message)
                else:
                    logger.warning(message)
        except Exception as e:
            logger.error(f"Error saat otomatisasi: {str(e)}")
    
    # Jalankan pemeriksaan dan sinkronisasi di thread terpisah
    auto_thread = threading.Thread(target=auto_check_and_sync)
    auto_thread.daemon = True
    auto_thread.start()
    
    return ui_components

# Fungsi _disable_ui_during_processing dan _cleanup_ui dipindahkan ke ui_helpers.py

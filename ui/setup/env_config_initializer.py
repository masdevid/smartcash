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
from smartcash.common.config.manager import ConfigManager
from smartcash.ui.utils.ui_logger import create_direct_ui_logger

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi environment
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Inisialisasi environment manager
    env_manager = get_environment_manager()
    
    # Inisialisasi config manager
    config_manager = ConfigManager.get_instance()
    
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

def _disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Nonaktifkan UI selama proses berjalan
    
    Args:
        ui_components: Dictionary berisi komponen UI
        disable: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang akan dinonaktifkan
    buttons = ['drive_button', 'directory_button', 'check_button', 'save_button']
    
    # Nonaktifkan atau aktifkan tombol
    for button_name in buttons:
        if button_name in ui_components:
            ui_components[button_name].disabled = disable

def _cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan UI setelah proses selesai
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Aktifkan kembali tombol
    _disable_ui_during_processing(ui_components, False)
    
    # Sembunyikan progress bar dan message
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    if 'progress_message' in ui_components:
        ui_components['progress_message'].layout.visibility = 'hidden'

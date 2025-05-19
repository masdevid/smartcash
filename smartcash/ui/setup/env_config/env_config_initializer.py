"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Initializer untuk konfigurasi environment dengan alur otomatis yang lebih robust
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.env_config.components.env_config_component import create_env_config_ui
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
from smartcash.ui.setup.env_config.handlers.auto_check_handler import setup_auto_check_handler
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.ui_logger import create_direct_ui_logger

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk konfigurasi environment.
    
    Returns:
        Dictionary UI components
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
    
    # Setup auto check handler
    setup_auto_check_handler(ui_components)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components

def _disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """
    Nonaktifkan UI selama proses berjalan.
    
    Args:
        ui_components: Dictionary UI components
        disable: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang akan dinonaktifkan
    button_keys = ['drive_button', 'directory_button', 'check_button', 'save_button']
    
    # Nonaktifkan tombol
    for key in button_keys:
        if key in ui_components:
            ui_components[key].disabled = disable

def _cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan UI setelah proses selesai.
    
    Args:
        ui_components: Dictionary UI components
    """
    # Aktifkan kembali tombol
    _disable_ui_during_processing(ui_components, False)
    
    # Sembunyikan progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Sembunyikan progress message
    if 'progress_message' in ui_components:
        ui_components['progress_message'].layout.visibility = 'hidden'

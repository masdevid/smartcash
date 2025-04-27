"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk modul konfigurasi environment dengan implementasi DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.setup.env_config_component import create_env_config_ui
from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk konfigurasi environment.
    
    Returns:
        Dictionary komponen UI yang terinisialisasi
    """
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['save_button', 'reset_button', 'check_button']
    
    # Gunakan base initializer
    return initialize_module_ui(
        module_name='env_config',
        create_ui_func=create_env_config_ui,
        setup_specific_handlers_func=setup_env_config_handlers,
        button_keys=button_keys
    )
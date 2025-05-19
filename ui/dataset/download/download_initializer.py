"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer untuk modul download dataset dengan integrasi progress tracking
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.download.components.download_component import create_download_ui
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers

def initialize_dataset_download_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk download dataset.
    
    Returns:
        Dictionary UI components yang terinisialisasi
    """
    # Gunakan base initializer dengan konfigurasi minimal
    return initialize_module_ui(
        module_name='download',
        create_ui_func=create_download_ui,
        setup_specific_handlers_func=setup_download_handlers,
        button_keys=['download_button', 'check_button', 'reset_button']
    )
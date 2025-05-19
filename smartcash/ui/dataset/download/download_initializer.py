"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer untuk modul download dataset dengan integrasi progress tracking
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.download.components.download_component import create_download_ui
from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
from smartcash.ui.dataset.download.handlers.api_key_handler import check_colab_secrets
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager

def initialize_dataset_download_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk download dataset.
    
    Returns:
        Dictionary UI components yang terinisialisasi
    """
    # Inisialisasi environment dan config manager
    env_manager = get_environment_manager()
    if not getattr(env_manager, 'base_dir', None):
        raise ValueError("base_dir must not be None. Please provide a valid base directory for configuration.")
    config_manager = get_config_manager(base_dir=env_manager.base_dir, config_file='dataset_config.yaml')
    
    # Gunakan base initializer dengan konfigurasi minimal
    ui_components = initialize_module_ui(
        module_name='download',
        create_ui_func=create_download_ui,
        # Catatan: setup_download_handlers sudah dipanggil di base_initializer.py
        # untuk module_name='download', jadi tidak perlu dipanggil lagi di sini
        button_keys=['download_button', 'check_button', 'reset_button', 'save_button', 'cleanup_button']
    )
    
    # Tambahkan config manager ke UI components
    ui_components['config_manager'] = config_manager
    
    # Periksa Colab secrets setelah UI diinisialisasi
    check_colab_secrets(ui_components)
    
    return ui_components
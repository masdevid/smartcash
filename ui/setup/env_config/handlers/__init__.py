"""
File: smartcash/ui/setup/env_config/handlers/__init__.py
Deskripsi: Package untuk handlers environment config
"""

# Import kelas-kelas handler baru
from smartcash.ui.setup.env_config.handlers.environment_setup_handler import EnvironmentSetupHandler
from smartcash.ui.setup.env_config.handlers.local_setup_handler import LocalSetupHandler
from smartcash.ui.setup.env_config.handlers.colab_setup_handler import ColabSetupHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler

# Import fungsi-fungsi untuk kompatibilitas mundur
from smartcash.ui.setup.env_config.handlers.config_info_handler import display_config_info
from smartcash.ui.setup.env_config.handlers.setup_handler import perform_setup, handle_setup_error
from smartcash.ui.setup.env_config.handlers.environment_setup_handler import setup_environment

__all__ = [
    # Kelas-kelas handler utama
    'EnvironmentSetupHandler',
    'LocalSetupHandler',
    'ColabSetupHandler',
    'SetupHandler',
    'EnvironmentHandler',
    'AutoCheckHandler',
    
    # Fungsi-fungsi untuk kompatibilitas mundur
    'display_config_info',
    'perform_setup',
    'handle_setup_error',
    'setup_environment'
]

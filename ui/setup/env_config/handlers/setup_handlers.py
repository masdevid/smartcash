"""
File: smartcash/ui/setup/env_config/handlers/setup_handlers.py
Deskripsi: Setup handlers untuk UI environment config
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.drive_handler import setup_drive_handler
from smartcash.ui.setup.env_config.handlers.directory_handler import setup_directory_handler

__all__ = [
    'setup_env_config_handlers',
    'setup_drive_handler',
    'setup_directory_handler',
]

def setup_env_config_handlers(ui_components: Dict[str, Any], colab_manager: Any) -> None:
    """
    Setup handlers untuk UI environment config
    
    Args:
        ui_components: Dictionary UI components
        colab_manager: ColabConfigManager instance
    """
    # Setup handlers
    setup_drive_handler(ui_components, colab_manager)
    setup_directory_handler(ui_components, colab_manager)

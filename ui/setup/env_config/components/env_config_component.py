"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component untuk konfigurasi environment
"""

from typing import Dict, Any
from IPython.display import display
import asyncio
from pathlib import Path
from functools import reduce

from smartcash.common.logger import get_logger

from smartcash.ui.setup.env_config.components.ui_creator import create_env_config_ui
from smartcash.ui.setup.env_config.components.manager_setup import setup_managers
from smartcash.ui.setup.env_config.components.progress_setup import setup_progress
from smartcash.ui.setup.env_config.handlers.setup_handlers import setup_env_config_handlers
from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler

logger = get_logger(__name__)

class EnvConfigComponent:
    """
    Component untuk konfigurasi environment
    """
    
    def __init__(self):
        """
        Inisialisasi component
        """
        self.logger = logger
        
        # Create UI first
        self.ui_components = create_env_config_ui()
        
        # Initialize managers
        self.config_manager, self.colab_manager, self.base_dir, self.config_dir = setup_managers()
        
        # Setup progress tracking
        setup_progress(self.ui_components)
        
        # Initialize handlers
        setup_env_config_handlers(self.ui_components, self.colab_manager)
        self.auto_check = AutoCheckHandler(self)
        
        # Run auto check without drive mounting
        asyncio.create_task(self.auto_check.auto_check())
    
    def _update_status(self, message: str, status_type: str = "info"):
        """
        Update the status panel with a message and type.
        Args:
            message: The message to display.
            status_type: The type of status (info, success, error).
        """
        from smartcash.ui.setup.env_config.utils.ui_utils import update_status
        update_status(self.ui_components, message, status_type)

    def display(self):
        """
        Display the environment configuration UI.
        """
        # Check if Drive is already connected and directories are set up
        if self.colab_manager.is_drive_connected():
            self.ui_components['drive_button'].disabled = True
            self._update_status("Google Drive already connected", "success")
        else:
            self.ui_components['drive_button'].disabled = False

        # Check if directories are already set up
        base_dirs = [
            "/content/data",
            "/content/models",
            "/content/output",
            "/content/logs",
            "/content/exports"
        ]
        all_dirs_exist = all(Path(dir_path).exists() for dir_path in base_dirs)
        if all_dirs_exist:
            self.ui_components['directory_button'].disabled = True
            self._update_status("Directories already set up", "success")
        else:
            self.ui_components['directory_button'].disabled = False

        # Display the UI components
        display(self.ui_components['drive_button'])
        display(self.ui_components['directory_button'])
        display(self.ui_components['status_panel'])
        if 'log_panel' in self.ui_components:
            display(self.ui_components['log_panel'])
        if 'progress_bar' in self.ui_components:
            display(self.ui_components['progress_bar'])
        if 'progress_message' in self.ui_components:
            display(self.ui_components['progress_message'])

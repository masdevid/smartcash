"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component UI untuk konfigurasi environment
"""

from typing import Dict, Any
from IPython.display import display
from pathlib import Path
import os

from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler
from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.utils.env_ui_utils import update_status, update_progress, log_message, MODULE_LOGGER_NAME
from smartcash.common.utils import is_colab

class EnvConfigComponent:
    """
    Component UI untuk konfigurasi environment
    """
    
    def __init__(self):
        """
        Inisialisasi component
        """
        
        # Create UI components first
        self.ui_components = UIFactory.create_ui_components()
        
        # Setup logger dengan namespace khusus environment config
        logger = create_ui_logger(self.ui_components, ENV_CONFIG_LOGGER_NAMESPACE)
        self.ui_components['logger'] = logger
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Setup UI callbacks for handlers
        ui_callbacks = {
            'log_message': self._log_message,
            'update_status': self._update_status,
            'update_progress': self._update_progress
        }
        
        # Initialize handlers with UI callbacks
        self.env_handler = EnvironmentHandler(ui_callbacks)
        self.auto_check_handler = AutoCheckHandler(ui_callbacks)
        
        # Connect button to setup function
        self.ui_components['setup_button'].on_click(self._handle_setup_click)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """
        Update the status panel with a message and type.
        Args:
            message: The message to display.
            status_type: The type of status (info, success, error).
        """
        update_status(self.ui_components, message, status_type)

    def _update_progress(self, value: float, message: str = ""):
        """
        Update progress bar and message
        """
        update_progress(self.ui_components, value, message)

    def _log_message(self, message: str, level: str = "info", icon: str = None):
        """
        Log a message to the output panel
        """
        log_message(self.ui_components, message, level, icon)

    def _handle_setup_click(self, button):
        """
        Handle setup button click
        """
        button.disabled = True
        
        try:
            # Delegate setup to environment handler
            success = self.env_handler.perform_setup()
            
            # Update button state based on result
            if success:
                button.disabled = True
            else:
                button.disabled = False
                
        except Exception as e:
            self._update_status(f"Error: {str(e)}", "error")
            self._log_message(f"Error: {str(e)}", "error", "❌")
            button.disabled = False

    def display(self):
        """
        Display the environment configuration UI.
        """
        # Check if required directories exist
        all_dirs_exist = self.env_handler.check_required_dirs()
        
        if all_dirs_exist:
            self.ui_components['setup_button'].disabled = True
            self._update_status("Environment sudah terkonfigurasi", "success")
            self._log_message("Environment sudah terkonfigurasi", "success", "✅")
            
            # Initialize config singleton if environment is already set up
            config_manager = self.env_handler.initialize_config_singleton()
            if config_manager:
                self.ui_components['config_manager'] = config_manager
        else:
            self.ui_components['setup_button'].disabled = False
            self._update_status("Environment perlu dikonfigurasi", "info")
            self._log_message("Environment perlu dikonfigurasi", "info", "ℹ️")

        # Display the UI components
        display(self.ui_components['ui_layout'])
        
        # Run auto check after UI is displayed
        self.auto_check_handler.check_environment() 
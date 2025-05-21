"""
File: smartcash/ui/components/env_config_component.py
Deskripsi: Component UI untuk konfigurasi environment
"""

from typing import Dict, Any
from IPython.display import display
from pathlib import Path
import os
import logging

from smartcash.ui.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.handlers.auto_check_handler import AutoCheckHandler
from smartcash.ui.setup.env_config.components.ui_creator import create_env_config_ui
from smartcash.ui.utils.env_ui_utils import update_status, update_progress, log_message
from smartcash.common.utils import is_colab

class EnvConfigComponent:
    """
    Component UI untuk konfigurasi environment
    """
    
    def __init__(self):
        """
        Inisialisasi component
        """
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Create UI components first
        self.ui_components = create_env_config_ui()
        
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

    def _log_message(self, message: str):
        """
        Log a message to the output panel
        """
        log_message(self.ui_components, message)

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
            self._log_message(f"Error: {str(e)}")
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
            
            # Initialize config singleton if environment is already set up
            self.env_handler.initialize_config_singleton()
        else:
            self.ui_components['setup_button'].disabled = False
            self._update_status("Environment perlu dikonfigurasi", "info")

        # Display the UI components
        display(self.ui_components['ui_layout'])
        
        # Run auto check after UI is displayed
        self.auto_check_handler.check_environment() 
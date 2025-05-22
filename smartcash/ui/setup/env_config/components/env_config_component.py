"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component UI untuk konfigurasi environment - diperbaiki dengan menghilangkan business logic dan memperbaiki logging
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.setup.env_config.handlers.auto_check_handler import AutoCheckHandler

class EnvConfigComponent:
    """
    Component UI untuk konfigurasi environment - fokus hanya pada UI coordination tanpa business logic
    """
    
    def __init__(self):
        """Inisialisasi component dengan proper separation of concerns"""
        
        # Create UI components first
        self.ui_components = UIFactory.create_ui_components()
        
        # Initialize handlers dengan UI components (bukan callbacks)
        self.env_handler = EnvironmentHandler(self.ui_components)
        self.auto_check_handler = AutoCheckHandler(self.ui_components)
        
        # Connect button click handler
        self.ui_components['setup_button'].on_click(self._handle_setup_click)
    
    def _handle_setup_click(self, button):
        """
        Handle setup button click - delegate ke handler tanpa business logic di UI
        """
        button.disabled = True
        
        try:
            # Delegate setup ke environment handler
            success = self.env_handler.perform_setup()
            
            # Update button state based on result
            button.disabled = success  # Keep disabled if success, enable if failed
                
        except Exception as e:
            # Log error melalui handler
            self.env_handler._log_message(f"Error saat setup: {str(e)}", "error", "❌")
            button.disabled = False

    def display(self):
        """
        Display the environment configuration UI dengan auto-check
        """
        # Display the UI components first
        display(self.ui_components['ui_layout'])
        
        # Run auto check untuk determine initial state
        try:
            env_status = self.auto_check_handler.check_environment()
            
            # Update button state berdasarkan hasil check
            if env_status and not env_status.get('error'):
                is_ready = (
                    not env_status.get('is_colab', False) or  # Not in Colab, or
                    not env_status.get('missing_dirs', [])    # No missing dirs
                )
                
                if is_ready:
                    self.ui_components['setup_button'].disabled = True
                    # Initialize config manager jika environment sudah ready
                    config_manager = self.env_handler.initialize_config_singleton()
                    if config_manager:
                        self.ui_components['config_manager'] = config_manager
                else:
                    self.ui_components['setup_button'].disabled = False
            else:
                self.ui_components['setup_button'].disabled = False
                
        except Exception as e:
            # Log error tapi jangan crash UI
            if hasattr(self, 'env_handler'):
                self.env_handler._log_message(f"Error saat auto-check: {str(e)}", "warning", "⚠️")


# Factory function untuk kompatibilitas
def create_env_config_component() -> EnvConfigComponent:
    """Factory function untuk membuat EnvConfigComponent"""
    return EnvConfigComponent()
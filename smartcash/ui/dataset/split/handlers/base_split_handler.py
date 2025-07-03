"""
File: smartcash/ui/dataset/split/handlers/base_split_handler.py
Deskripsi: Base handler untuk split module dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.base_handler import BaseHandler

class BaseSplitHandler(BaseHandler):
    """Base handler untuk split module dengan centralized error handling.
    
    Provides shared functionality for all split handlers:
    - Centralized logging dengan consistent module naming
    - Error handling integration dengan error_handler.py
    - Button state management (enable/disable)
    - UI component management helpers
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize base split handler dengan centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize with split module name
        super().__init__(module_name="split", **kwargs)
        
        # Store UI components reference
        self.ui_components = ui_components or {}
        
        # Setup log redirection if UI components are provided
        if ui_components and 'log_output' in ui_components:
            self.setup_log_redirection(ui_components)
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components reference for handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.ui_components = ui_components
        
        # Setup log redirection if log_output is available
        if 'log_output' in ui_components:
            self.setup_log_redirection(ui_components)
    
    def get_button_keys(self) -> list:
        """Get list of button keys for split module.
        
        Returns:
            List of button keys
        """
        return ['save_button', 'reset_button']
    
    def get_output_keys(self) -> list:
        """Get list of output keys for split module.
        
        Returns:
            List of output keys
        """
        return ['log_output']

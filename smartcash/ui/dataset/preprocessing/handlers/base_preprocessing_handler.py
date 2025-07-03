"""
File: smartcash/ui/dataset/preprocessing/handlers/base_preprocessing_handler.py
Deskripsi: Base handler untuk preprocessing module dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.base_handler import BaseHandler

class BasePreprocessingHandler(BaseHandler):
    """Base handler untuk preprocessing module dengan centralized error handling.
    
    Provides shared functionality for all preprocessing handlers:
    - Centralized logging dengan consistent module naming
    - Error handling integration dengan error_handler.py
    - Confirmation dialog utilities
    - Button state management (enable/disable)
    - Status panel update wrapper
    - UI component management helpers
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize base preprocessing handler dengan centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize with preprocessing module name
        super().__init__(module_name="preprocessing", **kwargs)
        
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
        """Get list of button keys for preprocessing module.
        
        Returns:
            List of button keys
        """
        return ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    
    def get_output_keys(self) -> list:
        """Get list of output keys for preprocessing module.
        
        Returns:
            List of output keys
        """
        return ['log_output', 'status_panel', 'summary_container']
    
    def get_progress_keys(self) -> list:
        """Get list of progress keys for preprocessing module.
        
        Returns:
            List of progress keys
        """
        return ['progress_tracker']
    
    def update_summary(self, message: str, status_type: str = 'info', title: str = None) -> None:
        """Update summary container with message.
        
        Args:
            message: Message to display
            status_type: Status type (info, success, warning, error)
            title: Optional title for summary
        """
        if not self.ui_components:
            self.logger.warning("⚠️ UI components tidak tersedia untuk update summary")
            return
            
        summary_container = self.ui_components.get('summary_container')
        if not summary_container:
            self.logger.warning("⚠️ Summary container tidak tersedia")
            return
            
        # Use parent class method for status panel update
        self.update_status_panel(summary_container, message, status_type, title)

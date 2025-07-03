"""
File: smartcash/ui/dataset/downloader/handlers/base_downloader_handler.py
Description: Base handler for downloader module with centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors

class BaseDownloaderHandler(BaseHandler):
    """Base handler for downloader module with centralized error handling."""
    
    @handle_ui_errors(error_component_title="Downloader Handler Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize base downloader handler with centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        # Set default module name and parent module if not provided
        module_name = kwargs.get('module_name', 'downloader')
        parent_module = kwargs.get('parent_module', 'dataset')
        
        # Initialize parent with proper module naming
        super().__init__(module_name=module_name, parent_module=parent_module)
        
        # Store UI components
        self.ui_components = ui_components or {}
        
    @handle_ui_errors(error_component_title="UI Operation Error", log_error=True)
    def update_ui_component(self, component_key: str, value: Any) -> bool:
        """Update a UI component with proper error handling.
        
        Args:
            component_key: Key of the component in ui_components
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        if component_key in self.ui_components:
            try:
                setattr(self.ui_components[component_key], 'value', value)
                return True
            except Exception as e:
                self.logger.error(f"Failed to update UI component {component_key}: {str(e)}")
                return False
        return False
        
    @handle_ui_errors(error_component_title="UI Access Error", log_error=True)
    def get_ui_component_value(self, component_key: str, default=None) -> Any:
        """Get a UI component value with proper error handling.
        
        Args:
            component_key: Key of the component in ui_components
            default: Default value if component not found or error occurs
            
        Returns:
            Component value or default
        """
        if component_key in self.ui_components:
            try:
                return getattr(self.ui_components[component_key], 'value', default)
            except Exception as e:
                self.logger.error(f"Failed to get UI component {component_key} value: {str(e)}")
                return default
        return default

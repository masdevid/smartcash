
"""
File: smartcash/ui/setup/dependency/handlers/config_handler.py
Description: Config handler for dependency management with centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler as BaseConfigHandler
from .base_dependency_handler import BaseDependencyHandler
from .config_extractor import extract_dependency_config
from .config_updater import update_dependency_ui
from .defaults import get_default_dependency_config

class DependencyConfigHandler(BaseDependencyHandler, BaseConfigHandler):
    """Config handler for dependency management with centralized error handling"""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize dependency config handler.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config from UI components.
        
        Args:
            ui_components: Dictionary containing UI components
            
        Returns:
            Dictionary containing extracted configuration
        """
        try:
            return extract_dependency_config(ui_components)
        except Exception as e:
            return self.handle_error(e, "Failed to extract dependency configuration")
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components from configuration.
        
        Args:
            ui_components: Dictionary containing UI components
            config: Configuration dictionary to apply
        """
        try:
            update_dependency_ui(ui_components, config)
        except Exception as e:
            self.handle_error(e, "Failed to update UI from configuration")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dependency configuration.
        
        Returns:
            Dictionary containing default configuration
        """
        try:
            return get_default_dependency_config()
        except Exception as e:
            return self.handle_error(e, "Failed to get default dependency configuration")
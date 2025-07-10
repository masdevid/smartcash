"""
File: smartcash/ui/dataset/visualization/configs/visualization_config_handler.py
Description: Configuration handler for the visualization module
"""

from typing import Dict, Any, Optional
from dataclasses import asdict, fields
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from .visualization_defaults import DEFAULT_CONFIG, VisualizationDefaults

class VisualizationConfigHandler(ConfigHandler):
    """Configuration handler for the visualization module."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the config handler.
        
        Args:
            config: Optional initial configuration
        """
        super().__init__(config or {})
        
        # Get default values from the VisualizationDefaults class
        self._defaults = {}
        for field in fields(VisualizationDefaults):
            if field.name != 'DEFAULT_CONFIG':
                self._defaults[field.name] = getattr(DEFAULT_CONFIG, field.name)
        
        # Set default values if not provided
        for key, value in self._defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            dict: Validated and normalized configuration
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Create a copy to avoid modifying the original
        validated = config.copy()
        
        # Get default values
        defaults = self.get_default_config()
        
        # Ensure required fields exist or use defaults
        for key, default_value in defaults.items():
            if key not in validated:
                validated[key] = default_value
                
        return validated
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration.
        
        Returns:
            dict: Default configuration
        """
        return self._defaults.copy()
    
    def update_from_ui(self, ui_components: Dict[str, Any]) -> None:
        """Update configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
        """
        # Update configuration from UI components
        for key, widget in ui_components.items():
            if hasattr(widget, 'value'):
                self._config[key] = widget.value
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components from configuration.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration to apply to the UI
        """
        # Update UI components from configuration
        for key, value in config.items():
            if key in ui_components:
                widget = ui_components[key]
                if hasattr(widget, 'value'):
                    try:
                        widget.value = value
                    except Exception as e:
                        self.logger.warning(f"Failed to update UI component {key}: {str(e)}")
    
    def update_ui(self, ui_components: Dict[str, Any]) -> None:
        """Update UI components from configuration.
        
        Args:
            ui_components: Dictionary of UI components
        """
        self.update_ui_from_config(ui_components, self._config)

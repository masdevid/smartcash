"""
File: smartcash/ui/dataset/visualization/configs/visualization_config_handler.py
Description: Configuration handler for the visualization module with shared config support
"""

from typing import Dict, Any, Optional
from dataclasses import asdict, fields
from smartcash.ui.core.handlers.config_handler import SharedConfigHandler
from smartcash.ui.logger import get_module_logger
from .visualization_defaults import DEFAULT_CONFIG, VisualizationDefaults

class VisualizationConfigHandler(SharedConfigHandler):
    """Configuration handler for the visualization module with shared config support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the config handler with shared config support.
        
        Args:
            config: Optional initial configuration
        """
        # Initialize logger first
        self.logger = get_module_logger("smartcash.ui.dataset.visualization.config")
        
        # Initialize defaults from DEFAULT_CONFIG
        self._defaults = {}
        if hasattr(DEFAULT_CONFIG, 'to_dict'):
            self._defaults = DEFAULT_CONFIG.to_dict()
        elif hasattr(DEFAULT_CONFIG, '__dict__'):
            self._defaults = {k: v for k, v in DEFAULT_CONFIG.__dict__.items()
                           if not k.startswith('_') and not callable(v)}
        
        # Initialize config with provided config or defaults
        if config is None:
            config = {}
        elif isinstance(config, str):
            # If config is a string, treat it as a config name and use defaults
            config_name = config
            config = self._defaults.copy()
            self.logger.info(f"Using default configuration for '{config_name}'")
        
        # Initialize with module name and parent module for shared config
        try:
            # Try to initialize with shared config if available
            if hasattr(super(), '__init__'):
                try:
                    super().__init__(
                        module_name='visualization',
                        parent_module='dataset',
                        default_config=config,
                        enable_sharing=True
                    )
                    # If we get here, the parent class initialized successfully
                    return
                except TypeError as e:
                    self.logger.debug(f"Falling back to basic initialization: {e}")
            
            # Fallback to basic initialization
            self._config = config.copy() if hasattr(config, 'copy') else dict(config)
            
        except Exception as e:
            self.logger.error(f"Error initializing config handler: {e}", exc_info=True)
            self._config = self._defaults.copy()
        
        # Ensure _config is a dictionary
        if not hasattr(self, '_config') or not isinstance(self._config, dict):
            self._config = {}
        
        # Get default values from the VisualizationDefaults class
        self._defaults = {}
        for field in fields(VisualizationDefaults):
            if field.name != 'DEFAULT_CONFIG':
                self._defaults[field.name] = getattr(DEFAULT_CONFIG, field.name)
        
        # Initialize with default values if not provided
        self._initialize_defaults()
        
        # Initialize shared configuration
        try:
            self.initialize()
        except Exception as e:
            self.logger.warning(f"Failed to initialize shared config: {e}")
            # Fall back to local config if shared config fails
            self._config = config.copy() if hasattr(config, 'copy') else dict(config)
    
    def _initialize_defaults(self):
        """Initialize default configuration values."""
        for key, value in self._defaults.items():
            if key not in self._config or self._config[key] is None:
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

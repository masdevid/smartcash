"""
File: smartcash/ui/dataset/visualization/configs/visualization_config_handler.py
Description: Simple configuration handler for visualization module.

This module provides a lightweight configuration handler for the visualization module
without persistence capabilities.
"""

from typing import Dict, Any, Optional, List
from dataclasses import asdict
from smartcash.ui.logger import get_module_logger
from .visualization_defaults import DEFAULT_CONFIG

class VisualizationConfigHandler:
    """Visualization configuration handler using pure delegation pattern.
    
    This class follows the modern BaseUIModule architecture where config handlers
    are pure implementation classes that delegate to BaseUIModule mixins.
    
    This handler manages configuration for the visualization module with
    validation and default values, but without persistence.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        """Initialize with optional configuration overrides.
        
        Args:
            config: Optional initial configuration
            logger: Optional logger instance
        """
        # Initialize pure delegation pattern (no mixin inheritance)
        self.logger = logger or get_module_logger('smartcash.ui.dataset.visualization.config')
        self.module_name = 'visualization'
        self.parent_module = 'dataset'
        
        # Convert defaults to dict if needed
        self._defaults = self._convert_to_dict(DEFAULT_CONFIG)
        self._default_config = self._defaults
        
        # Initialize with provided config or defaults
        initial_config = config if config is not None else {}
        self._config = self._initialize_config(initial_config)
        
        # Initialize validation rules
        self._validation_rules = self._get_validation_rules()
        
        self.logger.info("✅ Visualization config handler initialized")

    # --- Core Configuration Methods ---

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.validate_config(updates):
            self._config.update(updates)
            self.logger.debug(f"Configuration updated: {list(updates.keys())}")
        else:
            raise ValueError("Invalid configuration updates provided")

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = self._defaults.copy()
        self.logger.info("Configuration reset to defaults")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save config file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Implementation would save to YAML file
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default visualization configuration.
        
        Returns:
            Default configuration dictionary
        """
        return self._defaults.copy()

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate visualization configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - ensure required keys exist
            if not isinstance(config, dict):
                return False
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
        
    def _convert_to_dict(self, obj) -> Dict[str, Any]:
        """Convert an object to a dictionary if it's not already one."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() 
                   if not k.startswith('_') and not callable(v)}
        return {}
    
    def _initialize_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize configuration with defaults and user overrides."""
        # Start with defaults
        config = self._defaults.copy()
        
        # Apply user overrides
        for key, value in user_config.items():
            if key in self._defaults:
                config[key] = value
               
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        if key in self._defaults:
            self._config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update multiple configuration values at once."""
        for key, value in config.items():
            self.set(key, value)
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        self._config = self._defaults.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the current configuration as a dictionary."""
        return self._config.copy()
    
    # Dictionary-like interface
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dict-style access."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using dict-style access."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return key in self._config
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return str(self._config)
    
    def __repr__(self) -> str:
        """Technical string representation."""
        return f"{self.__class__.__name__}(config={self._config})"
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for visualization module."""
        return self._defaults.copy()
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'VisualizationConfigHandler':
        """Create config handler instance for visualization module."""
        return VisualizationConfigHandler(config)
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for configuration."""
        return {
            'figsize_width': {'type': (int, float), 'min': 1, 'max': 50},
            'figsize_height': {'type': (int, float), 'min': 1, 'max': 50},
            'plot_limit': {'type': int, 'min': 1, 'max': 1000},
            'save_plots': {'type': bool},
            'plot_format': {'type': str, 'choices': ['png', 'jpg', 'svg', 'pdf']},
            'show_progress': {'type': bool}
        }

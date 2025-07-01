"""
Base configuration mixin for environment configuration handlers.

This module provides a mixin class that enforces consistent configuration
handling across all environment configuration handlers.
"""
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path

class BaseConfigMixin:
    """Mixin class for consistent configuration handling.
    
    This mixin enforces a consistent pattern for configuration access and
    management across all environment configuration handlers.
    """
    
    # Default configuration for this handler
    DEFAULT_CONFIG: Dict[str, Any] = {}
    
    def __init__(self, *args, config_handler=None, **kwargs):
        """Initialize the configuration mixin.
        
        Args:
            config_handler: The configuration handler instance
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        
        # Store config handler reference
        self._config_handler = config_handler
        
        # Initialize config from handler or use defaults
        self._config = self._load_config()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            The current configuration dictionary
        """
        return self._config
    
    @property
    def config_handler(self):
        """Get the configuration handler.
        
        Returns:
            The configuration handler instance
            
        Raises:
            RuntimeError: If config handler is not set
        """
        if not self._config_handler:
            raise RuntimeError("Configuration handler not set")
        return self._config_handler
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from the config handler.
        
        Returns:
            The loaded configuration dictionary
            
        Raises:
            RuntimeError: If config handler is not available
        """
        if not self._config_handler:
            return self.DEFAULT_CONFIG.copy()
            
        # Get handler-specific config section
        handler_name = getattr(self, 'module_name', self.__class__.__name__.lower())
        return self._config_handler.get_handler_config(handler_name, self.DEFAULT_CONFIG)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update the configuration.
        
        Args:
            updates: Dictionary of configuration updates
            
        Raises:
            RuntimeError: If config handler is not available
        """
        if not self._config_handler:
            raise RuntimeError("Cannot update config: config handler not set")
            
        # Update local config
        self._config.update(updates)
        
        # Update config in handler
        handler_name = getattr(self, 'module_name', self.__class__.__name__.lower())
        self._config_handler.update_handler_config(handler_name, updates)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or default
        """
        return self._config.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.update_config({key: value})
    
    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = self.DEFAULT_CONFIG.copy()
        if self._config_handler:
            handler_name = getattr(self, 'module_name', self.__class__.__name__.lower())
            self._config_handler.reset_handler_config(handler_name, self.DEFAULT_CONFIG)

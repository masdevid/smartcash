"""
File: smartcash/ui/core/configs/base_config_handler.py
Base configuration handler for SmartCash UI modules.

Provides common functionality for managing module configurations,
including validation, defaults, and UI synchronization.
"""

from typing import Dict, Any, Optional, TypeVar, Generic, Type
from abc import ABC, abstractmethod
import logging

T = TypeVar('T')

class BaseConfigHandler(ABC):
    """Base class for configuration handlers in SmartCash UI modules.
    
    This class provides common functionality for managing module configurations,
    including validation, defaults, and UI synchronization.
    
    Subclasses should implement the abstract methods to provide module-specific
    configuration handling.
    """
    
    def __init__(self, 
                module_name: str, 
                default_config: Dict[str, Any],
                config_schema: Optional[Dict] = None):
        """Initialize the configuration handler.
        
        Args:
            module_name: Name of the module this handler manages configs for
            default_config: Default configuration values
            config_schema: Optional JSON schema for configuration validation
        """
        self.module_name = module_name
        self.default_config = default_config
        self.config_schema = config_schema
        self.logger = logging.getLogger(f"smartcash.ui.config.{module_name}")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Start with default config
        validated = self.default_config.copy()
        
        # Update with provided values
        if config:
            for key, value in config.items():
                if key in validated:
                    # Type checking and conversion could be added here
                    validated[key] = value
                else:
                    self.logger.warning(
                        f"Ignoring unknown config key '{key}' for module '{self.module_name}'"
                    )
        
        # Additional validation can be added in subclasses
        self._validate_config_impl(validated)
        
        return validated
    
    def _validate_config_impl(self, config: Dict[str, Any]) -> None:
        """Implementation-specific configuration validation.
        
        Subclasses can override this to add custom validation logic.
        By default, no additional validation is performed.
        
        Args:
            config: Configuration to validate (already normalized)
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def update_ui_from_config(self, 
                            ui_components: Dict[str, Any], 
                            config: Dict[str, Any]) -> None:
        """Update UI components based on configuration.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration to apply
            
        Raises:
            ValueError: If configuration is invalid for UI
        """
        pass
    
    @abstractmethod
    def get_ui_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing configuration from UI
            
        Raises:
            ValueError: If UI state is invalid
        """
        pass
    
    def merge_configs(self, 
                     base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration with values to override
            
        Returns:
            Merged configuration
        """
        result = base_config.copy()
        result.update(override_config)
        return result
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get a copy of the default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return self.default_config.copy()
    
    def __str__(self) -> str:
        """String representation of the config handler."""
        return f"{self.__class__.__name__}(module='{self.module_name}')"

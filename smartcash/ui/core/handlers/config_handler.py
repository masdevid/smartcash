# smartcash/ui/core/handlers/config_handler.py
"""
Config handler class for managing configuration in SmartCash UI.
Extends BaseHandler with configuration management capabilities.
"""
from typing import Dict, Any, Optional, Callable
import logging
import copy

from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.shared.logger import get_ui_logger

class ConfigHandler(BaseHandler):
    """
    Handler for managing configuration in SmartCash UI.
    
    This class extends BaseHandler with configuration management capabilities,
    including loading, saving, validating, and applying configurations.
    """
    
    def __init__(
        self,
        ui_components: Dict[str, Any],
        default_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the config handler.
        
        Args:
            ui_components: Dictionary containing UI components
            default_config: Default configuration dictionary
            logger: Optional logger instance
        """
        self.default_config = copy.deepcopy(default_config)
        self._config = copy.deepcopy(default_config)
        super().__init__(ui_components, logger)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return copy.deepcopy(self._config)
    
    def set_config(self, config: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """
        Set the configuration.
        
        Args:
            config: New configuration dictionary
            validate: Whether to validate the configuration before setting
            
        Returns:
            Dict with operation status and any error information
        """
        try:
            if validate and not self.validate_config(config):
                return {
                    "success": False,
                    "error": "Invalid configuration",
                    "handler": self.__class__.__name__
                }
            
            self._config = copy.deepcopy(config)
            self.update_ui_from_config()
            
            return {
                "success": True,
                "handler": self.__class__.__name__
            }
        except Exception as e:
            return self.handle_error(e, "Failed to set configuration")
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Returns:
            Configuration dictionary extracted from UI
        """
        # This method should be implemented by subclasses
        return copy.deepcopy(self._config)
    
    def update_ui_from_config(self) -> None:
        """Update UI components from the current configuration."""
        # This method should be implemented by subclasses
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the given configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # This method should be implemented by subclasses
        return True
    
    def reset_to_defaults(self) -> Dict[str, Any]:
        """
        Reset configuration to default values.
        
        Returns:
            Dict with operation status
        """
        try:
            self._config = copy.deepcopy(self.default_config)
            self.update_ui_from_config()
            
            return {
                "success": True,
                "handler": self.__class__.__name__
            }
        except Exception as e:
            return self.handle_error(e, "Failed to reset configuration")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        try:
            # Extract fresh config from UI to ensure it's up to date
            self._config = self.extract_config_from_ui()
            return self.config
        except Exception as e:
            self.logger.error(f"Error getting config: {str(e)}")
            return self.config  # Return last known good config

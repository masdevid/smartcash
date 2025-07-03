# smartcash/ui/core/handlers/ui_handler.py
"""
Module UI handler class for managing module-specific UI components and operations.
Extends BaseHandler with UI-specific capabilities.
"""
from typing import Dict, Any, Optional, Callable, Union, Type
import logging
import importlib
import traceback

from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.shared.logger import get_ui_logger
from smartcash.ui.utils.fallback_utils import create_fallback_ui, FallbackConfig
from smartcash.ui.decorators import safe_ui_operation
from smartcash.ui.components.error.error_component import create_error_component


class UIHandler(BaseHandler):
    """
    Handler for managing module-specific UI components and operations.
    
    This class extends BaseHandler with UI-specific capabilities, including
    dynamic loading of module-specific config components, UI updates, and
    error handling with proper UI feedback.
    """
    
    def __init__(
        self,
        ui_components: Dict[str, Any],
        parent_module: str,
        module_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the module UI handler.
        
        Args:
            ui_components: Dictionary containing UI components
            parent_module: Parent module name (e.g., 'dataset', 'setup')
            module_name: Module name (e.g., 'downloader', 'env_config')
            logger: Optional logger instance
        """
        self.parent_module = parent_module
        self.module_name = module_name
        self.logger = logger or get_ui_logger(
            module_name=module_name,
            parent_module=f"ui.{parent_module}"
        )
        
        # Initialize config components
        self._config_handler = None
        self._config_extractor = None
        self._config_updater = None
        self._config_validator = None
        self._config_defaults = None
        
        super().__init__(ui_components, self.logger)
    
    def setup(self) -> None:
        """Set up the handler after initialization."""
        try:
            self._load_config_components()
        except Exception as e:
            self.logger.error(f"Failed to load config components: {str(e)}")
    
    @safe_ui_operation(operation_name="load_config_components", log_level="error")
    def _load_config_components(self) -> None:
        """
        Dynamically load module-specific config components.
        
        This method attempts to load the following components:
        - Config handler
        - Config extractor
        - Config updater
        - Config validator
        - Config defaults
        """
        if not self.parent_module or not self.module_name:
            self.logger.warning("Parent module or module name not provided, skipping config component loading")
            return
        
        try:
            # Try to load config handler
            handler_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.configs.handler"
            handler_module = importlib.import_module(handler_path)
            if hasattr(handler_module, "get_config_handler"):
                self._config_handler = handler_module.get_config_handler(self.ui_components)
            
            # Try to load config extractor
            extractor_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.configs.extractor"
            extractor_module = importlib.import_module(extractor_path)
            if hasattr(extractor_module, "extract_config"):
                self._config_extractor = extractor_module.extract_config
            
            # Try to load config updater
            updater_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.configs.updater"
            updater_module = importlib.import_module(updater_path)
            if hasattr(updater_module, "update_ui"):
                self._config_updater = updater_module.update_ui
            
            # Try to load config validator
            validator_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.configs.validator"
            validator_module = importlib.import_module(validator_path)
            if hasattr(validator_module, "validate_config"):
                self._config_validator = validator_module.validate_config
            
            # Try to load config defaults
            defaults_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.configs.defaults"
            defaults_module = importlib.import_module(defaults_path)
            if hasattr(defaults_module, "DEFAULT_CONFIG"):
                self._config_defaults = defaults_module.DEFAULT_CONFIG
        
        except ImportError as e:
            self.logger.debug(f"Could not import config component: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading config components: {str(e)}")
    
    @safe_ui_operation(operation_name="extract_config", log_level="error")
    def extract_config(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Returns:
            Configuration dictionary extracted from UI
        """
        if self._config_handler and hasattr(self._config_handler, "extract_config_from_ui"):
            return self._config_handler.extract_config_from_ui()
        
        if self._config_extractor:
            return self._config_extractor(self.ui_components)
        
        self.logger.warning("No config extractor available")
        return {}
    
    @safe_ui_operation(operation_name="update_ui", log_level="error")
    def update_ui(self, config: Dict[str, Any]) -> None:
        """
        Update UI components from configuration.
        
        Args:
            config: Configuration dictionary
        """
        if self._config_handler and hasattr(self._config_handler, "update_ui_from_config"):
            self._config_handler.set_config(config, validate=False)
            return
        
        if self._config_updater:
            self._config_updater(self.ui_components, config)
            return
        
        self.logger.warning("No config updater available")
    
    @safe_ui_operation(operation_name="validate_config", log_level="error", fallback_return=False)
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if self._config_handler and hasattr(self._config_handler, "validate_config"):
            return self._config_handler.validate_config(config)
        
        if self._config_validator:
            return self._config_validator(config)
        
        self.logger.warning("No config validator available")
        return True
    
    @safe_ui_operation(operation_name="reset_config", log_level="error")
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Dict with operation status
        """
        if self._config_handler and hasattr(self._config_handler, "reset_to_defaults"):
            return self._config_handler.reset_to_defaults()
        
        if self._config_defaults and self._config_updater:
            self._config_updater(self.ui_components, self._config_defaults)
            return {"success": True, "handler": self.__class__.__name__}
        
        self.logger.warning("No config defaults or updater available")
        return {"success": False, "error": "No config defaults or updater available"}
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle errors with proper UI feedback.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
            
        Returns:
            Dict containing error information and UI components
        """
        error_message = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(f"Error in {self.__class__.__name__}: {error_message}")
        
        # Create error UI component
        error_ui = create_error_component(
            error_message=error_message,
            traceback=traceback.format_exc(),
            module_name=f"{self.parent_module}.{self.module_name}"
        )
        
        return {
            "success": False,
            "error": error_message,
            "error_type": type(error).__name__,
            "handler": self.__class__.__name__,
            "ui": error_ui
        }

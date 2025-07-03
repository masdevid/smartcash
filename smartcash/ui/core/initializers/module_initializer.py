# smartcash/ui/core/initializers/module_initializer.py
"""
Module initializer class for initializing module-specific UI components.
Extends BaseInitializer with module-specific capabilities.
"""
from typing import Dict, Any, Optional, Type
import logging
import importlib

from smartcash.ui.core.initializers.base_initializer import BaseInitializer
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.decorators import safe_ui_operation


class ModuleInitializer(BaseInitializer):
    """
    Initializer for module-specific UI components.
    
    This class extends BaseInitializer with module-specific capabilities,
    including handler setup and UI component management.
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the module initializer.
        
        Args:
            module_name: Name of the module being initialized
            parent_module: Parent module name (e.g., 'dataset', 'setup')
            logger: Optional logger instance
        """
        super().__init__(module_name, parent_module, logger)
        
        # Module-specific handlers
        self.ui_handler = None
        
        # Ensure UI components include log output/accordion if not already present
        if 'log_output' not in self.ui_components:
            self.ui_components['log_output'] = None
        if 'log_accordion' not in self.ui_components:
            self.ui_components['log_accordion'] = None
    
    @safe_ui_operation(operation_name="setup_handlers", log_level="error")
    def setup_handlers(self) -> Dict[str, Any]:
        """
        Set up handlers for UI components.
        
        Returns:
            Dict with setup status and any error information
        """
        try:
            # Set up UI handler
            self.ui_handler = self._create_ui_handler()
            
            # Set up module-specific handlers
            self._setup_module_handlers()
            
            return {
                "status": True,
                "initializer": self.__class__.__name__
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "initializer": self.__class__.__name__
            }
    
    def _create_ui_handler(self) -> ModuleUIHandler:
        """
        Create the UI handler for this module.
        
        Returns:
            ModuleUIHandler instance
        """
        # Try to dynamically load the module-specific handler class
        try:
            handler_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.handlers.{self.module_name}_handler"
            handler_module = importlib.import_module(handler_path)
            
            # Look for a class named [ModuleName]Handler (e.g., DownloaderHandler)
            handler_class_name = f"{self.module_name.capitalize()}Handler"
            if hasattr(handler_module, handler_class_name):
                handler_class = getattr(handler_module, handler_class_name)
                return handler_class(self.ui_components, self.parent_module, self.module_name, self.logger)
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not load module-specific handler: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error creating UI handler: {str(e)}")
        
        # Fall back to generic ModuleUIHandler
        return ModuleUIHandler(self.ui_components, self.parent_module, self.module_name, self.logger)
    
    def _setup_module_handlers(self) -> None:
        """
        Set up module-specific handlers.
        
        This method should be implemented by subclasses to set up
        any additional handlers needed for the module.
        """
        pass
    
    @safe_ui_operation(operation_name="update_ui_from_config", log_level="error")
    def update_ui_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update UI components from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict with update status and any error information
        """
        try:
            if self.ui_handler:
                self.ui_handler.update_ui(config)
                return {
                    "status": True,
                    "message": "UI updated successfully",
                    "initializer": self.__class__.__name__
                }
            
            return {
                "status": False,
                "message": "UI handler not initialized",
                "error": "UI handler not initialized",
                "initializer": self.__class__.__name__
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "initializer": self.__class__.__name__
            }
    
    @safe_ui_operation(operation_name="extract_config_from_ui", log_level="error")
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Returns:
            Configuration dictionary extracted from UI
        """
        if self.ui_handler:
            return self.ui_handler.extract_config()
        
        self.logger.warning("UI handler not initialized, cannot extract config")
        return {}
    
    @safe_ui_operation(operation_name="validate_config", log_level="error", fallback_return=False)
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if self.ui_handler:
            return self.ui_handler.validate_config(config)
        
        self.logger.warning("UI handler not initialized, cannot validate config")
        return False
    
    @safe_ui_operation(operation_name="reset_config", log_level="error")
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Dict with operation status
        """
        if self.ui_handler:
            return self.ui_handler.reset_config()
        
        return {
            "status": False,
            "message": "UI handler not initialized",
            "error": "UI handler not initialized",
            "initializer": self.__class__.__name__
        }

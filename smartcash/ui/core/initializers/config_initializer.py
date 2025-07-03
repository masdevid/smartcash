# smartcash/ui/core/initializers/config_initializer.py
"""
Config initializer class for initializing configuration-specific UI components.
Extends BaseInitializer with configuration-specific capabilities.
"""
from typing import Dict, Any, Optional, Callable, List, Union, Type
import logging
import importlib
import copy

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.shared.logger import get_ui_logger
from smartcash.ui.decorators import safe_ui_operation


class ConfigInitializer(ModuleInitializer):
    """
    Initializer for configuration-specific UI components.
    
    This class extends ModuleInitializer with configuration-specific capabilities,
    including config loading, saving, and validation.
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: str,
        default_config: Optional[Dict[str, Any]] = None,
        persistence_enabled: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the config initializer.
        
        Args:
            module_name: Name of the module being initialized
            parent_module: Parent module name (e.g., 'dataset', 'setup')
            default_config: Optional default configuration dictionary
            persistence_enabled: Whether to enable configuration persistence
            logger: Optional logger instance
        """
        super().__init__(module_name, parent_module, logger)
        
        # Configuration state
        self._default_config = copy.deepcopy(default_config) if default_config else {}
        self._config = copy.deepcopy(self._default_config)
        self.persistence_enabled = persistence_enabled
        
        # Config manager
        self.config_manager = None
        
        # Ensure UI components include log output/accordion if not already present
        if 'log_output' not in self.ui_components:
            self.ui_components['log_output'] = None
        if 'log_accordion' not in self.ui_components:
            self.ui_components['log_accordion'] = None
    
    def setup_handlers(self) -> Dict[str, Any]:
        """
        Set up handlers for UI components.
        
        Returns:
            Dict with setup status and any error information
        """
        # First set up the UI handler from the parent class
        result = super().setup_handlers()
        if not result.get("success", False):
            return result
        
        try:
            # Set up config manager if persistence is enabled
            if self.persistence_enabled:
                self._setup_config_manager()
            
            # Load initial config
            self._load_initial_config()
            
            return {
                "status": True,  # Using 'status' instead of 'success' for consistency
                "initializer": self.__class__.__name__
            }
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "initializer": self.__class__.__name__
            }
    
    def _setup_config_manager(self) -> None:
        """
        Set up the configuration manager.
        
        This method can be overridden by subclasses to set up
        a custom configuration manager.
        """
        # Try to dynamically load the shared config manager
        try:
            manager_path = "smartcash.ui.core.shared.shared_config_manager"
            manager_module = importlib.import_module(manager_path)
            
            if hasattr(manager_module, "SharedConfigManager"):
                manager_class = getattr(manager_module, "SharedConfigManager")
                self.config_manager = manager_class(
                    module_name=self.module_name,
                    parent_module=self.parent_module,
                    logger=self.logger
                )
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not load shared config manager: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error setting up config manager: {str(e)}")
    
    def _load_initial_config(self) -> None:
        """Load the initial configuration."""
        if self.persistence_enabled and self.config_manager:
            # Try to load config from persistent storage
            try:
                loaded_config = self.config_manager.load_config()
                if loaded_config:
                    self._config = loaded_config
                    self.update_ui_from_config(self._config)
                    return
            except Exception as e:
                self.logger.warning(f"Failed to load config from persistent storage: {str(e)}")
        
        # Fall back to default config
        self._config = copy.deepcopy(self._default_config)
        self.update_ui_from_config(self._config)
    
    @safe_ui_operation(operation_name="load_config", log_level="error")
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from persistent storage.
        
        Returns:
            Loaded configuration dictionary
        """
        if not self.persistence_enabled:
            return copy.deepcopy(self._config)
        
        if not self.config_manager:
            self.logger.warning("Config manager not initialized, returning in-memory config")
            return copy.deepcopy(self._config)
        
        try:
            loaded_config = self.config_manager.load_config()
            if loaded_config:
                self._config = loaded_config
                self.update_ui_from_config(self._config)
            return copy.deepcopy(self._config)
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            return copy.deepcopy(self._config)
    
    @safe_ui_operation(operation_name="save_config", log_level="error")
    def save_config(self) -> Dict[str, Any]:
        """
        Save configuration to persistent storage.
        
        Returns:
            Dict with save status and any error information
        """
        # Extract current config from UI
        current_config = self.extract_config_from_ui()
        
        # Validate config
        if not self.validate_config(current_config):
            return {
                "status": False,
                "error": "Invalid configuration",
                "initializer": self.__class__.__name__
            }
        
        # Update in-memory config
        self._config = current_config
        
        # Save to persistent storage if enabled
        if self.persistence_enabled and self.config_manager:
            try:
                self.config_manager.save_config(self._config)
                return {
                    "status": True,
                    "initializer": self.__class__.__name__
                }
            except Exception as e:
                return {
                    "status": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "initializer": self.__class__.__name__
                }
        
        # If persistence is disabled, just return success
        return {
            "status": True,
            "initializer": self.__class__.__name__
        }
    
    @safe_ui_operation(operation_name="reset_config", log_level="error")
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Dict with operation status
        """
        # Reset in-memory config
        self._config = copy.deepcopy(self._default_config)
        
        # Update UI
        self.update_ui_from_config(self._config)
        
        # Save to persistent storage if enabled
        if self.persistence_enabled and self.config_manager:
            try:
                self.config_manager.save_config(self._config)
            except Exception as e:
                return {
                    "status": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "initializer": self.__class__.__name__
                }
        
        return {
            "status": True,
            "initializer": self.__class__.__name__
        }
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return copy.deepcopy(self._config)
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """Get the default configuration."""
        return copy.deepcopy(self._default_config)

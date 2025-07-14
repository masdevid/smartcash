"""
Base UI Module class that integrates all common mixins.

This class provides a standard base for all UI modules with common functionality.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.core.mixins import (
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ProgressTrackingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin
)
from smartcash.ui.logger import get_module_logger


class BaseUIModule(
    ConfigurationMixin,
    OperationMixin,
    LoggingMixin,
    ProgressTrackingMixin,
    ButtonHandlerMixin,
    ValidationMixin,
    DisplayMixin,
    UIModule,
    ABC
):
    """
    Base UI Module class with all common functionality.
    
    This class combines all mixins to provide a comprehensive base for UI modules.
    Subclasses only need to implement module-specific functionality.
    """
    
    def __init__(self, module_name: str, parent_module: str = None):
        """
        Initialize base UI module.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
        """
        super().__init__(module_name, parent_module)
        
        # Set up logger
        self.logger = get_module_logger(f"smartcash.ui.{self.full_module_name}")
        
        # Initialize common attributes
        self._is_initialized = False
        self._ui_components = None
        self._required_components = []
        
        self.logger.debug(f"✅ BaseUIModule initialized: {self.full_module_name}")
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this module.
        
        Returns:
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def create_config_handler(self, config: Dict[str, Any]) -> Any:
        """
        Create config handler instance for this module.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Config handler instance
        """
        pass
    
    @abstractmethod
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create UI components for this module.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            UI components dictionary
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the module with all components.
        
        Returns:
            True if initialization was successful
        """
        try:
            if self._is_initialized:
                return True
            
            self.logger.debug(f"🔄 Initializing {self.full_module_name}")
            
            # Initialize configuration handler
            self._initialize_config_handler()
            
            # Create UI components
            self._ui_components = self.create_ui_components(self.get_current_config())
            
            # Initialize operation manager if method exists
            if hasattr(self, '_initialize_operation_manager'):
                self._initialize_operation_manager()
            
            # Setup button handlers
            self._setup_button_handlers()
            
            # Setup UI logging bridge
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    self._setup_ui_logging_bridge(operation_container)
            
            # Initialize progress display
            self._initialize_progress_display()
            
            # Register default operation handlers
            self._register_default_operations()
            
            self._is_initialized = True
            
            self.logger.info(f"✅ {self.full_module_name} initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {self.full_module_name}: {e}")
            return False
    
    def _register_default_operations(self) -> None:
        """Register default operation handlers."""
        # Register common operations
        self.register_operation_handler('save_config', self.save_config)
        self.register_operation_handler('reset_config', self.reset_config)
        self.register_operation_handler('get_status', self.get_status)
        self.register_operation_handler('validate_all', self.validate_all)
        self.register_operation_handler('display_ui', self.display_ui)
        
        # Register button handlers
        self.register_button_handler('save', lambda _: self.save_config())
        self.register_button_handler('reset', lambda _: self.reset_config())
    
    def get_module_info(self) -> Dict[str, Any]:
        """
        Get comprehensive module information.
        
        Returns:
            Module information dictionary
        """
        return {
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'full_module_name': self.full_module_name,
            'is_initialized': self._is_initialized,
            'has_config_handler': self._config_handler is not None,
            'has_operation_manager': getattr(self, '_operation_manager', None) is not None,
            'has_ui_components': self._ui_components is not None,
            'ui_components_count': len(self._ui_components) if self._ui_components else 0,
            'registered_operations': len(self._operation_handlers),
            'registered_button_handlers': len(self._button_handlers),
            'validation_errors': self.get_validation_errors(),
            'display_state': self.get_display_state(),
            'progress_state': self.get_progress_state()
        }
    
    def __repr__(self) -> str:
        """String representation of the module."""
        status = "initialized" if self._is_initialized else "not initialized"
        return f"BaseUIModule({self.full_module_name}, {status})"
    
    def __str__(self) -> str:
        """String representation of the module."""
        return f"{self.full_module_name} UI Module"
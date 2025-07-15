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
            
            # Validate button-handler integrity
            self._validate_button_handler_integrity()
            
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
    
    def _validate_button_handler_integrity(self) -> None:
        """
        Validate button-handler integrity during module setup.
        
        This method ensures that all buttons have corresponding handlers
        and follows naming conventions. It can auto-fix common issues.
        """
        try:
            from smartcash.ui.core.validation.button_validator import validate_button_handlers
            
            # Validate with auto-fix enabled
            result = validate_button_handlers(self, auto_fix=True)
            
            # Log validation results
            if result.has_errors:
                self.logger.error(f"Button validation errors in {self.full_module_name}: {len([i for i in result.issues if i.level.value == 'error'])} errors")
                for issue in result.issues:
                    if issue.level.value == 'error':
                        self.logger.error(f"🔘 {issue.message}")
                        if issue.suggestion:
                            self.logger.error(f"   💡 Suggestion: {issue.suggestion}")
            
            if result.has_warnings:
                self.logger.warning(f"Button validation warnings in {self.full_module_name}: {len([i for i in result.issues if i.level.value == 'warning'])} warnings")
                for issue in result.issues:
                    if issue.level.value == 'warning':
                        self.logger.warning(f"🔘 {issue.message}")
                        if issue.suggestion:
                            self.logger.warning(f"   💡 Suggestion: {issue.suggestion}")
            
            # Log auto-fixes
            if result.auto_fixes_applied:
                self.logger.info(f"Button validation auto-fixes applied: {len(result.auto_fixes_applied)}")
                for fix in result.auto_fixes_applied:
                    self.logger.info(f"🔧 {fix}")
            
            # Log success if no issues
            if result.is_valid and not result.has_warnings:
                self.logger.debug(f"✅ Button validation passed for {self.full_module_name}")
            
        except Exception as e:
            self.logger.warning(f"Button validation failed: {e}")
    
    def get_button_validation_status(self) -> Dict[str, Any]:
        """
        Get current button validation status.
        
        Returns:
            Dictionary with validation status information
        """
        try:
            from smartcash.ui.core.validation.button_validator import validate_button_handlers
            
            result = validate_button_handlers(self, auto_fix=False)
            
            return {
                'is_valid': result.is_valid,
                'has_errors': result.has_errors,
                'has_warnings': result.has_warnings,
                'error_count': len([i for i in result.issues if i.level.value == 'error']),
                'warning_count': len([i for i in result.issues if i.level.value == 'warning']),
                'button_count': len(result.button_ids),
                'handler_count': len(result.handler_ids),
                'missing_handlers': result.missing_handlers,
                'orphaned_handlers': result.orphaned_handlers,
                'issues': [
                    {
                        'level': issue.level.value,
                        'message': issue.message,
                        'button_id': issue.button_id,
                        'suggestion': issue.suggestion,
                        'auto_fixable': issue.auto_fixable
                    }
                    for issue in result.issues
                ]
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e)
            }
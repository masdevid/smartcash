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
from typing import Optional, Dict, Any


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
    
    Environment Features:
    - Set enable_environment=True to enable environment detection and management
    - Access environment paths via self.environment_paths
    - Check env status with self.is_colab and self.is_drive_mounted
    """
    
    @property
    def has_environment_support(self) -> bool:
        """Check if environment features are enabled."""
        return self._enable_environment and hasattr(self, '_environment_paths')
    
    @property
    def environment_paths(self) -> Optional[Dict[str, str]]:
        """Get environment paths if environment support is enabled."""
        if not self.has_environment_support:
            self.logger.warning("Environment support is not enabled for this module")
            return None
        return getattr(self, '_environment_paths', None)
    
    @property
    def is_colab(self) -> Optional[bool]:
        """Check if running in Google Colab (if environment support is enabled)."""
        return getattr(self, '_is_colab', None) if self.has_environment_support else None
    
    @property
    def is_drive_mounted(self) -> Optional[bool]:
        """Check if Google Drive is mounted (if environment support is enabled)."""
        return getattr(self, '_is_drive_mounted', None) if self.has_environment_support else None
    
    def __init__(self, module_name: str, parent_module: str = None, enable_environment: bool = False, **kwargs):
        """
        Initialize base UI module.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
            enable_environment: Whether to enable environment management features
            **kwargs: Additional keyword arguments for parent classes
        """
        # Initialize base UIModule first
        super().__init__(module_name, parent_module, **kwargs)
        
        # Initialize environment support if enabled
        self._enable_environment = enable_environment
        if self._enable_environment:
            from smartcash.ui.core.mixins.environment_mixin import EnvironmentMixin
            EnvironmentMixin.__init__(self)
        
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
            
            # Register default operation handlers FIRST
            self._register_default_operations()
            
            # Setup button handlers AFTER registration
            self._setup_button_handlers()
            
            # Validate button-handler integrity AFTER setup
            self._validate_button_handler_integrity()
            
            # Setup UI logging bridge
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    self._setup_ui_logging_bridge(operation_container)
            
            # Initialize progress display
            self._initialize_progress_display()
            
            # Flush any buffered logs now that operation container is ready
            if hasattr(self, '_flush_log_buffer'):
                self._flush_log_buffer()
            
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
        
        # Register button handlers with status updates
        self.register_button_handler('save', self._handle_save_config)
        self.register_button_handler('reset', self._handle_reset_config)
    
    def _handle_save_config(self, button=None) -> Dict[str, Any]:
        """Handle save config button click with status updates."""
        try:
            self.log("💾 Save config button clicked", 'info')
            
            result = self.save_config()
            if result.get('success'):
                success_msg = result.get('message', 'Configuration saved successfully')
                self.log(f"✅ {success_msg}", 'success')
                self.update_operation_status(success_msg, "success")
                self._update_header_status(success_msg, "success")
            else:
                error_msg = result.get('message', 'Save failed')
                self.log(f"❌ {error_msg}", 'error')
                self.update_operation_status(error_msg, "error")
                self._update_header_status(error_msg, "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Save config error: {e}"
            self.log(f"❌ {error_msg}", 'error')
            self.update_operation_status(error_msg, "error")
            self._update_header_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_reset_config(self, button=None) -> Dict[str, Any]:
        """Handle reset config button click with status updates."""
        try:
            self.log("🔄 Reset config button clicked", 'info')
            
            result = self.reset_config()
            if result.get('success'):
                success_msg = result.get('message', 'Configuration reset to defaults')
                self.log(f"✅ {success_msg}", 'success')
                self.update_operation_status(success_msg, "success")
                self._update_header_status(success_msg, "success")
            else:
                error_msg = result.get('message', 'Reset failed')
                self.log(f"❌ {error_msg}", 'error')
                self.update_operation_status(error_msg, "error")
                self._update_header_status(error_msg, "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Reset config error: {e}"
            self.log(f"❌ {error_msg}", 'error')
            self.update_operation_status(error_msg, "error")
            self._update_header_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
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
    
    def _update_header_status(self, message: str, status_type: str = "info") -> None:
        """
        Update header container status.
        
        Args:
            message: Status message
            status_type: Status type (info, success, warning, error)
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                header_container = self._ui_components.get('header_container')
                if header_container:
                    # Handle object-style header container
                    if hasattr(header_container, 'update_status'):
                        header_container.update_status(message, status_type)
                        return
                    # Handle dict-style header container
                    elif isinstance(header_container, dict) and 'update_status' in header_container:
                        header_container['update_status'](message, status_type)
                        return
                        
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update header status: {e}")
    
    def _execute_operation_with_wrapper(self, 
                                       operation_name: str,
                                       operation_func: callable,
                                       button: Any = None,
                                       validation_func: callable = None,
                                       success_message: str = None,
                                       error_message: str = None) -> Dict[str, Any]:
        """
        Common wrapper for executing operations with standard logging, progress, and button management.
        
        Args:
            operation_name: Display name for the operation
            operation_func: Function to execute the actual operation
            button: Button that triggered the operation
            validation_func: Optional validation function to run before operation
            success_message: Custom success message (optional)
            error_message: Custom error message prefix (optional)
            
        Returns:
            Operation result dictionary
        """
        # Extract button ID for individual button management
        button_id = getattr(button, 'description', operation_name.lower()).lower().replace(' ', '_')
        
        try:
            # Start operation logging and progress tracking
            self.log_operation_start(operation_name)
            self.start_progress(f"Memulai {operation_name.lower()}...", 0)
            self.update_operation_status(f"Memulai {operation_name.lower()}...", "info")
            
            # Only disable the clicked button, not all buttons
            self.disable_all_buttons(f"⏳ {operation_name}...", button_id=button_id)
            
            # Run validation if provided
            if validation_func:
                validation_result = validation_func()
                if not validation_result.get('valid', True):
                    warning_msg = validation_result.get('message', f'Validasi {operation_name.lower()} gagal')
                    self.log(f"⚠️ {warning_msg}", 'warning')
                    self.update_operation_status(warning_msg, "warning")
                    self.error_progress(warning_msg)
                    return {'success': False, 'message': warning_msg}
            
            # Update progress for execution
            self.update_progress(25, f"Memproses {operation_name.lower()}...")
            
            # Execute the actual operation
            result = operation_func()
            
            # Handle result
            if result.get('success'):
                success_msg = success_message or result.get('message', f'{operation_name} berhasil diselesaikan')
                self.log_operation_complete(operation_name)
                self.update_operation_status(success_msg, "success")
                self.log(f"✅ {success_msg}", 'success')
                self.complete_progress(f"{operation_name} selesai")
            else:
                error_prefix = error_message or f'{operation_name} gagal'
                error_msg = result.get('message', 'Operasi gagal')
                full_error = f"{error_prefix}: {error_msg}"
                self.log_operation_error(operation_name, error_msg)
                self.update_operation_status(full_error, "error")
                self.error_progress(full_error)
                
            return result
            
        except Exception as e:
            error_prefix = error_message or f'Kesalahan {operation_name.lower()}'
            error_msg = f"{error_prefix}: {e}"
            self.log_operation_error(operation_name, str(e))
            self.update_operation_status(error_msg, "error")
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable only the specific button that was disabled
            if button_id:
                self.enable_all_buttons(button_id=button_id)
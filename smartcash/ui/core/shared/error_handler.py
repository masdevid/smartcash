"""
File: smartcash/ui/core/shared/error_handler.py
Deskripsi: Centralized error handler untuk SmartCash UI Core menggunakan error components
"""

import traceback
from enum import Enum
from typing import Dict, Any, Optional, Callable, Type
from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.components.error import ErrorComponent, create_error_component

class ErrorLevel(Enum):
    """Error level enumeration."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class CoreErrorHandler:
    """Centralized error handler untuk UI Core dengan fail-fast principle."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        from smartcash.ui.core.shared.logger import get_enhanced_logger
        self.logger = get_enhanced_logger(f"smartcash.ui.core.{module_name}")
        self._error_count = 0
        self._last_error = None
        
    def handle_error(self, error_msg: str, level: ErrorLevel = ErrorLevel.ERROR, 
                    exc_info: bool = False, fail_fast: bool = True, 
                    create_ui_error: bool = False, **kwargs) -> Optional[Any]:
        """Handle error dengan centralized logging dan fail-fast option.
        
        Args:
            error_msg: Error message
            level: Error level
            exc_info: Include exception info
            fail_fast: Whether to raise exception (default: True)
            create_ui_error: Create UI error component
            **kwargs: Additional context
            
        Returns:
            UI error component jika create_ui_error=True, None otherwise
        """
        self._error_count += 1
        self._last_error = error_msg
        
        # Log berdasarkan level
        context = f" | Context: {kwargs}" if kwargs else ""
        full_msg = f"❌ {error_msg}{context}"
        
        if level == ErrorLevel.CRITICAL:
            self.logger.critical(full_msg, exc_info=exc_info)
        elif level == ErrorLevel.ERROR:
            self.logger.error(full_msg, exc_info=exc_info)
        elif level == ErrorLevel.WARNING:
            self.logger.warning(full_msg, exc_info=exc_info)
        else:
            self.logger.info(full_msg, exc_info=exc_info)
        
        # Create UI error component jika diminta
        ui_component = None
        if create_ui_error:
            ui_component = self._create_ui_error(error_msg, level, exc_info)
        
        # Fail-fast jika diminta
        if fail_fast and level in [ErrorLevel.CRITICAL, ErrorLevel.ERROR]:
            raise RuntimeError(f"[{self.module_name}] {error_msg}")
        
        return ui_component
    
    def _create_ui_error(self, error_msg: str, level: ErrorLevel, 
                        exc_info: bool = False) -> Any:
        """Create UI error component menggunakan error components."""
        try:
            # Get traceback jika exc_info
            traceback_text = None
            if exc_info:
                traceback_text = traceback.format_exc()
            
            # Map error level to error type
            error_type = level.value
            
            # Create error component
            error_component = create_error_component(
                title=f"🚨 {level.value.title()} - {self.module_name}",
                message=error_msg,
                traceback=traceback_text,
                error_type=error_type,
                show_details=True
            )
            
            return error_component
            
        except Exception as e:
            # Fallback jika error component gagal dibuat
            self.logger.error(f"Failed to create UI error component: {str(e)}")
            return None
    
    def handle_exception(self, exc: Exception, context: str = "", 
                        fail_fast: bool = True, create_ui_error: bool = False) -> Optional[Any]:
        """Handle exception dengan detailed logging dan UI error."""
        error_msg = f"Exception in {context}: {str(exc)}"
        return self.handle_error(
            error_msg, 
            ErrorLevel.ERROR, 
            exc_info=True, 
            fail_fast=fail_fast,
            create_ui_error=create_ui_error
        )
    
    def handle_validation_error(self, validation_msg: str, 
                               create_ui_error: bool = False, **kwargs) -> Optional[Any]:
        """Handle validation error dengan fail-fast."""
        return self.handle_error(
            f"Validation failed: {validation_msg}", 
            ErrorLevel.ERROR, 
            fail_fast=True,
            create_ui_error=create_ui_error,
            **kwargs
        )
    
    def handle_component_error(self, component_name: str, operation: str, 
                              error: Exception, fail_fast: bool = True,
                              create_ui_error: bool = False) -> Optional[Any]:
        """Handle component-specific error."""
        error_msg = f"Component '{component_name}' failed during {operation}: {str(error)}"
        return self.handle_error(
            error_msg, 
            ErrorLevel.ERROR, 
            exc_info=True, 
            fail_fast=fail_fast,
            create_ui_error=create_ui_error,
            component=component_name, 
            operation=operation
        )
    
    def handle_silent_error(self, error_msg: str, create_ui_error: bool = False, 
                           **kwargs) -> Optional[Any]:
        """Handle error dengan silent fail (hanya log warning)."""
        return self.handle_error(
            error_msg, 
            ErrorLevel.WARNING, 
            fail_fast=False,
            create_ui_error=create_ui_error,
            **kwargs
        )
    
    @property
    def error_count(self) -> int:
        return self._error_count
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    def reset_error_state(self) -> None:
        """Reset error state."""
        self._error_count = 0
        self._last_error = None
        self.logger.debug(f"🔄 Reset error state for {self.module_name}")

# Global error handlers by module
_error_handlers: Dict[str, CoreErrorHandler] = {}

def get_error_handler(module_name: str) -> CoreErrorHandler:
    """Get or create error handler for module."""
    if module_name not in _error_handlers:
        _error_handlers[module_name] = CoreErrorHandler(module_name)
    return _error_handlers[module_name]

def handle_component_validation(component_name: str, component: Any, 
                              required_attributes: list = None, 
                              module_name: str = "unknown",
                              create_ui_error: bool = False) -> Optional[Any]:
    """Validate component dengan fail-fast dan optional UI error."""
    error_handler = get_error_handler(module_name)
    
    if component is None:
        return error_handler.handle_validation_error(
            f"Component '{component_name}' is None",
            create_ui_error=create_ui_error
        )
    
    if required_attributes:
        for attr in required_attributes:
            if not hasattr(component, attr):
                return error_handler.handle_validation_error(
                    f"Component '{component_name}' missing required attribute '{attr}'",
                    create_ui_error=create_ui_error
                )
    
    return None

def safe_component_operation(component_name: str, operation: Callable, 
                           fail_fast: bool = True, module_name: str = "unknown",
                           create_ui_error: bool = False) -> Any:
    """Execute component operation dengan error handling."""
    error_handler = get_error_handler(module_name)
    
    try:
        return operation()
    except Exception as e:
        error_handler.handle_component_error(
            component_name, 
            "operation", 
            e, 
            fail_fast,
            create_ui_error=create_ui_error
        )
        return None

def validate_ui_components(ui_components: Dict[str, Any], 
                         required_components: list = None,
                         module_name: str = "unknown",
                         create_ui_error: bool = False) -> Optional[Any]:
    """Validate UI components dictionary dengan fail-fast."""
    error_handler = get_error_handler(module_name)
    
    if not ui_components:
        return error_handler.handle_validation_error(
            "UI components dictionary is empty or None",
            create_ui_error=create_ui_error
        )
    
    if required_components:
        for component_name in required_components:
            if component_name not in ui_components:
                return error_handler.handle_validation_error(
                    f"Required component '{component_name}' not found in UI components",
                    create_ui_error=create_ui_error
                )
            
            component = ui_components[component_name]
            if component is None:
                return error_handler.handle_validation_error(
                    f"Required component '{component_name}' is None",
                    create_ui_error=create_ui_error
                )
    
    return None

def handle_errors(module_name: str = "unknown", fail_fast: bool = True,
                 create_ui_error: bool = False):
    """Decorator untuk automatic error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler(module_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_exception(
                    e, 
                    f"function {func.__name__}", 
                    fail_fast,
                    create_ui_error=create_ui_error
                )
                return None
        return wrapper
    return decorator

class ErrorContext:
    """Context manager untuk error handling dengan UI error support."""
    
    def __init__(self, module_name: str, operation: str, fail_fast: bool = True,
                 create_ui_error: bool = False):
        self.error_handler = get_error_handler(module_name)
        self.operation = operation
        self.fail_fast = fail_fast
        self.create_ui_error = create_ui_error
        self.ui_error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.ui_error = self.error_handler.handle_exception(
                exc_val, 
                self.operation, 
                self.fail_fast,
                create_ui_error=self.create_ui_error
            )
        return False  # Don't suppress exceptions
    
    def get_ui_error(self) -> Optional[Any]:
        """Get created UI error component."""
        return self.ui_error
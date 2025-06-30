"""
File: smartcash/ui/utils/error_utils.py
Deskripsi: Consolidated error handling utilities for SmartCash UI components.

This module provides a comprehensive set of utilities for error handling, logging,
and user feedback in the SmartCash UI. It combines the best features from the
previous error_handler.py and error_utils.py modules.
"""

from typing import Any, Callable, Dict, Optional, TypeVar, Union, Type, Tuple
from functools import wraps
import traceback
from smartcash.ui.utils.ui_logger import UILogger, get_logger, get_module_logger
import warnings
from dataclasses import asdict

from smartcash.common.exceptions import ErrorContext, SmartCashError, UIError

T = TypeVar('T')

class ErrorHandler:
    """Centralized error handling for UI components with instance-based configuration.
    
    This class provides a more object-oriented approach to error handling while
    maintaining compatibility with the functional utilities in this module.
    """
    
    def __init__(
        self, 
        logger: Optional[UILogger] = None, 
        default_component: str = "ui",
        ui_components: Optional[Dict[str, Any]] = None
    ):
        """Initialize the error handler.
        
        Args:
            logger: UILogger instance for error logging
            default_component: Default component name for error context
            ui_components: Dictionary of UI components for error display
        """
        self.logger = logger or get_module_logger(__name__)
        self.default_component = default_component
        self.ui_components = ui_components or {}
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        ui_components: Optional[Dict[str, Any]] = None,
        show_ui: bool = True,
        log_level: str = "error"
    ) -> None:
        """Handle an error with proper logging and UI feedback.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
            ui_components: Dictionary of UI components for showing errors
            show_ui: Whether to show the error in the UI
            log_level: Log level ('error', 'warning', 'info', 'debug')
        """
        # Ensure we have a context
        if context is None:
            context = ErrorContext(component=self.default_component)
        
        # Log the error
        self._log_error(error, context, log_level)
        
        # Show error in UI if requested and components are available
        ui_comps = ui_components or self.ui_components
        if show_ui and ui_comps:
            self._show_ui_error(error, context, ui_comps)
    
    def _log_error(
        self,
        error: Exception,
        context: ErrorContext,
        log_level: str
    ) -> None:
        """Log the error with context information."""
        # Prepare log message
        log_method = getattr(self.logger, log_level.lower(), self.logger.error)
        
        # Format error details
        error_details = {
            'error_type': error.__class__.__name__,
            'message': str(error),
            'context': asdict(context)
        }
        
        # Add stack trace for non-SmartCash errors
        if not isinstance(error, SmartCashError):
            error_details['stack_trace'] = traceback.format_exc()
        
        # Log the error
        log_method("Error occurred", extra=error_details)
    
    def _show_ui_error(
        self,
        error: Exception,
        context: ErrorContext,
        ui_components: Dict[str, Any]
    ) -> None:
        """Display error in the UI."""
        try:
            # Get UI components with fallbacks
            status_panel = ui_components.get('status_panel')
            log_output = ui_components.get('log_output')
            
            # Format error message
            error_message = str(error)
            if context.component:
                error_message = f"[{context.component.upper()}] {error_message}"
            
            # Update status panel if available
            if hasattr(status_panel, 'update_status'):
                status_panel.update_status(error_message, 'error')
            
            # Log to UI console if available
            if hasattr(log_output, 'append_stdout'):
                log_output.append_stdout(f"{error_message}\n")
                
                # Add context details if available
                if context.details:
                    details = ", ".join(f"{k}: {v}" for k, v in context.details.items())
                    log_output.append_stdout(f"  Details: {details}\n")
                
                # Add stack trace for non-SmartCash errors
                if not isinstance(error, SmartCashError):
                    log_output.append_stderr(f"\nStack trace:\n{traceback.format_exc()}")
                        
        except Exception as e:
            # If we can't show the error in the UI, at least log it
            self.logger.error(
                "Failed to display error in UI",
                extra={'error': str(e), 'original_error': str(error)}
            )
    
    def wrap_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for async functions to handle errors consistently."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.handle_error(e)
                raise
        return wrapper
    
    def wrap_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for sync functions to handle errors consistently."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle_error(e)
                raise
        return wrapper

def create_error_context(
    component: str = "",
    operation: str = "",
    details: Optional[Dict[str, Any]] = None,
    ui_components: Optional[Dict[str, Any]] = None
) -> ErrorContext:
    """Create an ErrorContext with standardized parameter handling.
    
    Args:
        component: Component name where the error occurred
        operation: Operation being performed when error occurred
        details: Additional error details as key-value pairs
        ui_components: UI components for error display (deprecated, pass to handle_error instead)
        
    Returns:
        ErrorContext: Configured error context object
        
    Example:
        >>> ctx = create_error_context(
        ...     component="data_loader",
        ...     operation="load_dataset",
        ...     details={"dataset": "sample.csv", "attempt": 3}
        ... )
    """
    if ui_components is not None:
        warnings.warn(
            "The 'ui_components' parameter is deprecated. "
            "Pass UI components to handle_error() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
    return ErrorContext(
        component=component,
        operation=operation,
        details=details or {}
    )

def error_handler_scope(
    component: str = "ui",
    operation: str = "unknown",
    logger: Optional[UILogger] = None,
    ui_components: Optional[Dict[str, Any]] = None,
    show_ui_error: bool = True,
    log_level: str = "error"
):
    """Context manager for scoped error handling with automatic error logging and UI feedback.
    
    This context manager catches exceptions, logs them with the provided logger, and
    optionally displays them in the UI. It suppresses exceptions by default to allow
    execution to continue.
    
    Args:
        component: Name of the component where the error occurred
        operation: Name of the operation being performed
        logger: UILogger instance for error logging (default: module logger)
        ui_components: Dictionary of UI components for error display
        show_ui_error: Whether to show errors in the UI (if components are provided)
        log_level: Log level to use ('error', 'warning', 'info', 'debug')
        
    Returns:
        A context manager that handles errors within its scope
        
    Example:
        with error_handler_scope(component="data_processor", operation="process_data"):
            # Code that might raise exceptions
            result = process_data()
    """
    class ErrorScope:
        def __init__(self):
            self.context = create_error_context(
                component=component,
                operation=operation
            )
            self.logger = logger or get_module_logger(component)
            self.ui_components = ui_components or {}
            self.show_ui_error = show_ui_error
            self.log_level = log_level.lower()
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_value, exc_traceback):
            if exc_type is not None:
                # Create error handler for this scope
                handler = ErrorHandler(
                    logger=self.logger,
                    default_component=component,
                    ui_components=self.ui_components
                )
                
                # Create error context
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    details={
                        'exception_type': exc_type.__name__,
                        'exception_msg': str(exc_value)
                    }
                )
                
                # Handle the error (logs and optionally shows UI)
                handler.handle_error(
                    error=exc_value,
                    context=context,
                    show_ui=self.show_ui_error,
                    log_level=self.log_level
                )
                
                # Suppress the exception to continue execution
                return True
            return False
    
    return ErrorScope()

def with_error_handling(
    error_handler: Optional[ErrorHandler] = None,
    component: str = "ui",
    operation: str = "unknown",
    show_traceback: bool = False,
    ui_components: Optional[Dict[str, Any]] = None,
    fallback_value: Any = None,
    log_level: str = "error"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that wraps a function with error handling logic.
    
    This decorator catches exceptions, logs them, and optionally displays them in the UI.
    It can either use a provided ErrorHandler instance or create a default one.
    
    Args:
        error_handler: Optional ErrorHandler instance to use
        component: Name of the component where the function is defined
        operation: Name of the operation being performed
        show_traceback: Whether to include full traceback in logs
        ui_components: UI components for error display (if no error_handler provided)
        fallback_value: Value to return if an exception occurs
        log_level: Log level to use ('error', 'warning', 'info', 'debug')
        
    Returns:
        A decorator that wraps the function with error handling
        
    Example:
        @with_error_handling(component="data_processor", operation="process_data")
        def process_data():
            # Function that might raise exceptions
            return complex_operation()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create a default error handler if none provided
            handler = error_handler or ErrorHandler(
                logger=get_module_logger(component),
                default_component=component,
                ui_components=ui_components or {}
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context with function details
                error_context = create_error_context(
                    component=component,
                    operation=operation,
                    details={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                # Handle the error
                handler.handle_error(
                    error=e,
                    context=error_context,
                    show_ui=ui_components is not None,
                    log_level=log_level
                )
                
                # Return fallback value if provided, otherwise re-raise
                if fallback_value is not None:
                    return fallback_value
                raise
                
        return wrapper
    return decorator

def _fallback_error_handling(
    error: Exception, 
    context: ErrorContext, 
    show_traceback: bool = False
) -> None:
    """
    Fallback error handling ketika error handler tidak tersedia
    
    Args:
        error: Exception yang terjadi
        context: Error context
        show_traceback: Tampilkan traceback
    """
    # Format error message dengan context
    error_msg = f"🚨 [{context.component}:{context.operation}] {str(error)}"
    
    # Print error ke console
    print(error_msg)
    
    # Print traceback jika diminta
    if show_traceback:
        print("📋 Traceback:")
        traceback.print_exc()
    
    # Show UI error jika UI components tersedia
    if context.ui_components:
        try:
            from smartcash.ui.utils.fallback_utils import show_error_ui
            show_error_ui(context.ui_components, str(error))
        except Exception:
            pass  # Silent fail untuk UI error

def log_errors(
    logger: Optional[UILogger] = None,
    level: str = "error",
    component: str = "ui",
    operation: str = "unknown"
) -> Callable:
    """
    Simple error logging decorator with UI integration
    
    Args:
        logger: UILogger instance
        level: Log level (error, warning, info)
        component: Component name
        operation: Operation name
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = logger or get_module_logger(component)
                log_func = getattr(logger, level, logger.error)
                log_func(
                    f"Error in {component}.{operation}: {str(e)}",
                    extra={
                        'component': component,
                        'operation': operation,
                        'traceback': traceback.format_exc()
                    }
                )
                raise
        return wrapper
    return decorator

# FIXED: Safe error context creation tanpa parameter conflicts
def safe_create_context(comp: str, op: str, **kwargs) -> ErrorContext:
    """Safe error context creation dengan parameter aliases"""
    return create_error_context(
        component=comp,
        operation=op,
        details=kwargs.get('details'),
        ui_components=kwargs.get('ui_components')
    )

# One-liner utilities for common error patterns
handle_ui_error = lambda ui_components, error_msg: show_error_ui(ui_components, error_msg) if ui_components else get_module_logger().error(error_msg)
log_and_ignore = lambda logger, error, msg="": (logger or get_module_logger()).error(f"{msg}: {str(error)}")
safe_execute = lambda func, fallback=None: fallback if _safe_call(func, fallback) is None else _safe_call(func, fallback)

def _safe_call(func: Callable, fallback: Any = None) -> Any:
    """Safe function call dengan fallback"""
    try:
        return func()
    except Exception:
        return fallback

def safe_ui_operation(component: str = "ui", operation: str = "unknown"):
    """
    Decorator untuk menjalankan operasi UI dengan error handling yang aman
    
    Args:
        component: Nama komponen yang menggunakan decorator
        operation: Nama operasi yang di-wrap
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(__name__)
                error_msg = f"Error in {component}.{operation}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Get UI components from args or kwargs
                ui_components = kwargs.get('ui_components') or \
                              next((arg for arg in args if isinstance(arg, dict) and 'logger_bridge' in arg), None)
                
                if ui_components and 'logger_bridge' in ui_components:
                    ui_components['logger_bridge'].error(error_msg)
                
                # Return None or re-raise based on context
                if kwargs.get('silent', False):
                    return None
                raise
        return wrapper
    return decorator

# Export public API
__all__ = [
    # Core error handling
    'ErrorHandler',
    'create_error_context',
    'error_handler_scope',
    'with_error_handling',
    
    # Decorators
    'log_errors',
    'safe_ui_operation',
    
    # Utility functions
    'handle_ui_error',
    'log_and_ignore',
    'safe_execute',
    'safe_create_context',
    
    # For backward compatibility
    'safe_create_context',
]
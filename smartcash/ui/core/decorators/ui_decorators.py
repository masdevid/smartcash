"""
File: smartcash/ui/core/decorators/ui_decorators.py
Description: Utility decorators for UI operations using centralized error handling.

These decorators provide a clean way to handle common UI operation patterns with
consistent error handling, logging, and fallbacks using the centralized error
handling system.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps, partial

from smartcash.ui.core.errors import (
    ErrorLevel,
    get_error_handler,
    safe_component_operation as core_safe_component_operation,
    with_error_handling as core_with_error_handling
)

T = TypeVar('T')

# Re-export core error handling decorators for convenience
safe_component_operation = core_safe_component_operation
with_error_handling = core_with_error_handling

def safe_ui_operation(operation_name: str = "ui_operation", 
                     level: ErrorLevel = ErrorLevel.ERROR,
                     fallback_return: Any = None):
    """Decorator for safely executing UI operations with centralized error handling.
    
    This decorator wraps a function with error handling using the centralized
    error handling system, providing consistent error reporting and logging.
    
    Args:
        operation_name: Name of the operation for error context
        level: Error severity level (default: ErrorLevel.ERROR)
        fallback_return: Value to return if an exception occurs
        
    Returns:
        Decorated function that safely handles exceptions
    """
    def decorator(func):
        @wraps(func)
        @core_with_error_handling(
            error_message=f"Error in UI operation: {operation_name}",
            level=level,
            reraise=False,
            default=fallback_return
        )
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def safe_widget_operation(widget_key_arg: int = 0, widget_key_kwarg: str = "ui_components"):
    """Decorator for safely performing operations on UI widgets.
    
    This decorator handles operations on UI widgets safely using the centralized
    error handling system, with proper error context for widget operations.
    
    Args:
        widget_key_arg: Position of the UI components dictionary in args
        widget_key_kwarg: Name of the UI components keyword argument
        
    Returns:
        Decorated function that safely handles widget operations
    """
    def decorator(func):
        @wraps(func)
        @core_with_error_handling(
            error_message=lambda e, *a, **kw: (
                f"Error in widget operation {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            reraise=False,
            default=None
        )
        def wrapper(self, *args, **kwargs):
            # Get UI components from args or kwargs
            ui_components = None
            if len(args) > widget_key_arg:
                ui_components = args[widget_key_arg]
            elif widget_key_kwarg in kwargs:
                ui_components = kwargs[widget_key_kwarg]
            
            if not ui_components or not isinstance(ui_components, dict):
                get_error_handler().handle_warning(
                    f"No valid UI components for {func.__name__}",
                    context={
                        'function': func.__name__,
                        'widget_key_arg': widget_key_arg,
                        'widget_key_kwarg': widget_key_kwarg,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                return None
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def safe_progress_operation(progress_key: str = "progress_tracker", 
                           fallback_keys: List[str] = None):
    """Decorator for safely performing operations on progress trackers.
    
    This decorator handles progress tracker operations safely using the
    centralized error handling system, with fallbacks to standard progress bars
    if the tracker is not available.
    
    Args:
        progress_key: Key for the progress tracker in UI components
        fallback_keys: List of fallback keys to try if progress_key is not available
        
    Returns:
        Decorated function that safely handles progress operations
    """
    if fallback_keys is None:
        fallback_keys = ["progress_bar", "progress_container"]
    
    def decorator(func):
        @wraps(func)
        @core_safe_component_operation(
            error_message=lambda e, *a, **kw: (
                f"Error in progress operation {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            default=None
        )
        def wrapper(self, *args, **kwargs):
            # Add progress context for better error reporting
            context = {
                'progress_key': progress_key,
                'fallback_keys': fallback_keys,
                'function': func.__name__
            }
            
            # Set context for error reporting
            error_handler = get_error_handler()
            with error_handler.context(**context):
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator

def safe_component_access(component_type: str = None, default_value: Any = None):
    """Decorator for safely accessing and using UI components.
    
    This decorator provides safe access to UI components using the centralized
    error handling system, with proper context about the component being accessed.
    
    Args:
        component_type: Type of component being accessed (for error context)
        default_value: Default value to return if component access fails
        
    Returns:
        Decorated function that safely handles component access
    """
    def decorator(func):
        @wraps(func)
        @core_safe_component_operation(
            error_message=lambda e, *a, **kw: (
                f"Error accessing component {component_type or 'unknown'} in {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            default=default_value
        )
        def wrapper(self, *args, **kwargs):
            # Add component context for better error reporting
            context = {
                'component_type': component_type,
                'function': func.__name__,
                'default_value': default_value
            }
            
            # Set context for error reporting
            error_handler = get_error_handler()
            with error_handler.context(**context):
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator

"""
File: smartcash/ui/config_cell/handlers/error_handler.py

Error handling for config cell components using the consolidated utilities.

This module provides error handling decorators and utilities that integrate with
the centralized error handling and logging infrastructure.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Callable, Union
from functools import wraps

from smartcash.common.exceptions import SmartCashError
from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.utils.error_utils import ErrorHandler, ErrorContext
from smartcash.ui.utils.fallback_utils import (
    FallbackConfig,
    create_fallback_ui,
    safe_execute
)

# Get module logger
logger = get_module_logger(__name__)
T = TypeVar('T')

def handle_ui_errors(
    error_component_title: str = "Error",
    log_error: bool = True,
    return_type: Type[T] = dict,
    **error_handler_kwargs
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for handling errors in UI component methods.
    
    Args:
        error_component_title: Title for the error component
        log_error: Whether to log the error
        return_type: Expected return type of the decorated function
        **error_handler_kwargs: Additional kwargs for ErrorHandler
        
    Returns:
        A decorator that wraps the function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create error context with function name
            ctx = ErrorContext(
                component=func.__name__,
                operation="execution",
                **error_handler_kwargs
            )
            
            # Use ErrorHandler for consistent error handling
            handler = ErrorHandler(
                context=ctx,
                logger=logger,
                log_level='error' if log_error else None
            )
            
            # Execute with error handling
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                handler.handle_error(
                    error=e,
                    context=ErrorContext(
                        component=func.__name__,
                        operation="execution"
                    )
                )
                # Call the error handler's callback if provided
                return create_error_response(
                    error_message=str(e),
                    error=e,
                    title=error_component_title,
                    return_type=return_type
                )
            
            return result
                
        return wrapper
    return decorator

def create_error_response(
    error_message: str,
    error: Optional[Exception] = None,
    title: str = "Error",
    include_traceback: bool = True,
    return_type: Type[T] = dict,
    **fallback_kwargs
) -> Union[Dict[str, Any], T]:
    """Create an error response using the standard error component.
    
    Args:
        error_message: Error message to display
        error: Optional exception instance for traceback
        title: Title for the error component
        include_traceback: Whether to include traceback
        return_type: Type of the return value (dict or widget)
        **fallback_kwargs: Additional kwargs for FallbackConfig
        
    Returns:
        Either a dictionary with error info or a widget, based on return_type
    """
    # Ensure error_message is always a string
    if not isinstance(error_message, str):
        error_message = str(error_message) if error_message is not None else "An unknown error occurred"
    
    # If error is provided but error_message is generic, use error's string representation
    if error is not None and error_message == "An unknown error occurred":
        error_message = str(error)
    
    # Create fallback UI using the centralized utility
    error_ui = safe_execute(
        create_fallback_ui,
        error_message=error_message,
        exc_info=(type(error), error, error.__traceback__) if error else None,
        config=FallbackConfig(
            title=title,
            show_traceback=include_traceback,
            error_type="error",
            **fallback_kwargs
        )
    )
    
    # Return based on requested type
    if return_type == dict:
        return {
            'container': error_ui['ui'],
            'error': True,
            'error_details': error_ui.get('error_details', {})
        }
    return error_ui['ui']
"""
File: smartcash/ui/handlers/error_handler.py

⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.
Please update your imports to use smartcash.ui.core.errors instead.

Error handling for UI components using the consolidated utilities.

This module provides error handling decorators and utilities that integrate with
the centralized error handling and logging infrastructure.
"""
import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Callable, Union
from functools import wraps

# Issue deprecation warning
warnings.warn(
    "The 'smartcash.ui.handlers.error_handler' module is deprecated and will be removed in a future version. "
    "Please update your imports to use 'smartcash.ui.core.errors' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from smartcash.common.exceptions import SmartCashError
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors import (
    ErrorContext,
    create_error_component,
    handle_errors as core_handle_ui_errors,
    ErrorComponent
)
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.core.errors.enums import ErrorLevel

# Create a wrapper for create_error_response using create_error_component
def core_create_error_response(
    error_message: str,
    error: Optional[Exception] = None,
    title: str = "Error",
    include_traceback: bool = True,
    return_type: type = dict,
    **kwargs
):
    """Create an error response using the core error component."""
    error_component = create_error_component(
        message=error_message,
        error=error,
        title=title,
        include_traceback=include_traceback,
        **kwargs
    )
    
    if return_type == dict:
        return {
            'success': False,
            'error': error_message,
            'component': error_component
        }
    return error_component
from smartcash.ui.utils.fallback_utils import (
    FallbackConfig,
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
    """
    ⚠️ DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use smartcash.ui.core.errors.handle_errors instead.
    
    This is a compatibility wrapper around the core handle_errors decorator.
    """
    warnings.warn(
        "The 'handle_ui_errors' function is deprecated and will be removed in a future version. "
        "Please use 'smartcash.ui.core.errors.handle_errors' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Map the deprecated parameters to the new ones
    level = ErrorLevel.ERROR if log_error else ErrorLevel.DEBUG
    
    # Create a wrapper function that handles the return type conversion
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply the core error handler
        wrapped_func = core_handle_ui_errors(
            error_msg=error_component_title,
            level=level,
            reraise=False,  # We'll handle the return value
            log_error=log_error,
            create_ui=True,
            **error_handler_kwargs
        )(func)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                result = wrapped_func(*args, **kwargs)
                # If we got here, the function succeeded
                if return_type == dict and not isinstance(result, dict):
                    return {'success': True, 'result': result}  # type: ignore
                return result
            except Exception as e:
                error_message = str(e)
                if log_error:
                    logger.error(f"Error in {func.__name__}: {error_message}", exc_info=True)
                
                if return_type == dict:
                    return core_create_error_response(
                        error_message=error_message,
                        error=e,
                        title=error_component_title,
                        include_traceback=True,
                        return_type=dict,
                        **error_handler_kwargs
                    )
                else:
                    # For non-dict return types, re-raise the exception
                    raise
        
        return wrapper
    
    return decorator

def create_error_response(
    error_message: str,
    error: Optional[Exception] = None,
    title: str = "Error",
    include_traceback: bool = True,
    return_type: Type[T] = dict,
    **fallback_kwargs
) -> Union[Dict[str, Any], Any]:
    """
    ⚠️ DEPRECATED: This function is deprecated and will be removed in a future version.
    Please use smartcash.ui.core.errors.create_error_response instead.
    """
    warnings.warn(
        "The 'create_error_response' function is deprecated and will be removed in a future version. "
        "Please use 'smartcash.ui.core.errors.create_error_response' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Forward to the new implementation
    return core_create_error_response(
        error_message=error_message,
        error=error,
        title=title,
        include_traceback=include_traceback,
        return_type=return_type,
        **fallback_kwargs
    )
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
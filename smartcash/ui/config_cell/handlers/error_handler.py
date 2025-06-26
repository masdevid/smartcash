"""
Error handling for config cell components.

This module provides centralized error handling for UI components in the config cell.
It builds upon the base error utilities and exceptions.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from functools import wraps
import traceback
import logging

from IPython import display

from smartcash.common.exceptions import SmartCashError
from smartcash.ui.config_cell.utils.error_utils import create_error_fallback

logger = logging.getLogger(__name__)
T = TypeVar('T')

def handle_ui_errors(
    error_component_title: str = "Error",
    log_error: bool = True,
    return_type: Type[T] = dict
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to handle errors in UI component methods.
    
    Args:
        error_component_title: Title to use for error components
        log_error: Whether to log the error
        return_type: The expected return type of the decorated function
        
    Returns:
        A decorator that wraps the function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except SmartCashError as e:
                # For known errors, use the error message directly
                error_message = str(e)
                error_traceback = traceback.format_exc()
                logger.error(error_message, exc_info=True)
                
                error_ui = create_error_fallback(
                    error_message=error_message,
                    traceback=error_traceback if log_error else None
                )
                
                if return_type == dict:
                    return {'container': error_ui['container'], 'error': True}
                return error_ui
                
            except Exception as e:
                # For unexpected errors, include more details
                error_message = f"Unexpected error: {str(e)}"
                error_traceback = traceback.format_exc()
                logger.error(error_message, exc_info=True)
                
                error_ui = create_error_fallback(
                    error_message=error_message,
                    traceback=error_traceback
                )
                
                if return_type == dict:
                    return {'container': error_ui['container'], 'error': True}
                return error_ui
                
        return wrapper
    return decorator

def create_error_response(
    error_message: str,
    error: Optional[Exception] = None,
    title: str = "Error",
    include_traceback: bool = True
) -> Any:
    """
    Create a standardized error response with an error UI component.
    
    Args:
        error_message: The main error message to display
        error: Optional exception object
        title: Title for the error component
        include_traceback: Whether to include traceback in the error UI
        
    Returns:
        A widget containing the error UI
    """
    error_traceback = traceback.format_exc() if (error and include_traceback) else None
    
    if error:
        logger.error(
            f"{title}: {error_message}",
            exc_info=error if include_traceback else None
        )
    
    error_ui = create_error_fallback(
        error_message=error_message,
        traceback=error_traceback,
        title=title  # Pass the title to the error fallback
    )
    
    # Return the container widget directly
    if hasattr(error_ui, 'keys') and 'container' in error_ui:
        return error_ui['container']
    return error_ui
    
    # Fallback in case the container is not available
    from IPython.display import HTML
    return HTML(f"<div style='color:red;'>{title}: {error_message}</div>")


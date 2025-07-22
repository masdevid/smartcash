"""
Error handling utilities for SmartCash UI Core.

This module provides shared error handling functionality to break circular imports.
"""
import functools
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from .errors.enums import ErrorLevel
from .errors.error_component import create_error_component

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def create_default_error_handler():
    """Create a default error handler instance."""
    from .errors.handlers import CoreErrorHandler
    return CoreErrorHandler()

def with_error_handling(
    func: Optional[F] = None,
    error_message: Optional[Union[str, Callable[..., str]]] = None,
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = False,
    default: Any = None,
    **error_kwargs
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for adding error handling to a function.
    
    Args:
        func: The function to decorate (used internally by the decorator)
        error_message: Error message string or callable that generates the message
        level: Error severity level
        reraise: Whether to re-raise the exception after handling
        default: Default value to return if an exception occurs and reraise is False
        **error_kwargs: Additional keyword arguments to pass to the error handler
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                # Get the error message
                if callable(error_message):
                    msg = error_message(e, *args, **kwargs)
                else:
                    msg = error_message or f"Error in {f.__name__}: {str(e)}"
                
                # Get the error handler from instance or create default
                error_handler = getattr(args[0], 'error_handler', None) if args else None
                if not error_handler or not hasattr(error_handler, 'handle_error'):
                    error_handler = create_default_error_handler()
                
                # Handle the error
                error_handler.handle_error(
                    msg,
                    level=level,
                    exc_info=True,
                    **error_kwargs
                )
                
                if reraise:
                    raise
                return default
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            try:
                return await f(*args, **kwargs)
            except Exception as e:
                # Get the error message
                if callable(error_message):
                    msg = error_message(e, *args, **kwargs)
                else:
                    msg = error_message or f"Error in async {f.__name__}: {str(e)}"
                
                # Get the error handler from instance or create default
                error_handler = getattr(args[0], 'error_handler', None) if args else None
                if not error_handler or not hasattr(error_handler, 'handle_error'):
                    error_handler = create_default_error_handler()
                
                # Handle the error
                error_handler.handle_error(
                    msg,
                    level=level,
                    exc_info=True,
                    **error_kwargs
                )
                
                if reraise:
                    raise
                return default
        
        # Return the appropriate wrapper based on whether the function is async
        if inspect.iscoroutinefunction(f):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)
    
    # Handle the case when the decorator is used with parameters
    if func is None:
        return decorator
    return decorator(func)

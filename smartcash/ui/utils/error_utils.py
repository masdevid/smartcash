"""
File: smartcash/ui/utils/error_utils.py
Deskripsi: Utility functions and context managers for error handling
"""
from contextlib import contextmanager
from typing import Any, Dict, Optional, Type, TypeVar, Callable

from smartcash.common.exceptions import ErrorContext, SmartCashError
from smartcash.ui.utils.error_handler import ErrorHandler

T = TypeVar('T')

def create_error_context(
    component: str = "",
    operation: str = "",
    **details: Any
) -> ErrorContext:
    """Create an error context with the given information.
    
    Args:
        component: The component where the error occurred
        operation: The operation being performed
        **details: Additional context details
        
    Returns:
        ErrorContext: A new error context
    """
    return ErrorContext(
        component=component,
        operation=operation,
        details=details or None
    )

@contextmanager
def error_handler_scope(
    error_handler: ErrorHandler,
    component: str = "",
    operation: str = "",
    **details: Any
):
    """Context manager that provides error handling for a block of code.
    
    Example:
        with error_handler_scope(error_handler, component="preprocessing", operation="normalize_data") as context:
            # Your code here
            if something_wrong:
                raise ValueError("Something went wrong")
    """
    context = create_error_context(component, operation, **details)
    
    try:
        yield context
    except Exception as e:
        error_handler.handle_error(e, context=context)
        raise

def with_error_handling(
    error_handler: ErrorHandler,
    default_component: str = "",
    default_operation: str = "",
    **default_details: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add error handling to a function.
    
    Example:
        @with_error_handling(error_handler, component="preprocessing")
        def process_data(data):
            # Your code here
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            # Get component and operation from function if not provided
            component = default_component or func.__module__.split('.')[-1]
            operation = default_operation or func.__name__
            
            # Create context with any additional details
            context = create_error_context(
                component=component,
                operation=operation,
                **default_details
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context=context)
                raise
        
        return wrapper
    return decorator

def log_errors(error_handler: ErrorHandler, **context_kwargs):
    """Decorator that logs errors but doesn't suppress them."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, **context_kwargs)
                raise
        return wrapper
    return decorator

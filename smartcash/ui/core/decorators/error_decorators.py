"""
Centralized Error Handling Decorators

This module consolidates all error handling decorators used across SmartCash,
eliminating duplication and providing consistent error handling patterns.
"""

import functools
import logging
import inspect
import traceback
from dataclasses import asdict
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast, Union, List

from smartcash.ui.core.error_utils import with_error_handling
from smartcash.ui.core.errors.context import ErrorContext
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.common.exceptions import ErrorContext as CommonErrorContext

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

def handle_errors(
    error_msg: Optional[str] = None,
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = True,
    log_error: bool = True,
    create_ui: bool = False,
    handler: Optional[Any] = None,
    context_attr: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to handle errors in a consistent way.
    
    Args:
        error_msg: Custom error message to use. If None, the function's docstring
                  or a default message will be used.
        level: The error level to use for logging.
        reraise: Whether to re-raise the exception after handling.
        log_error: Whether to log the error.
        create_ui: Whether to create a UI error component.
        handler: Optional error handler instance to use.
        context_attr: Attribute name to get error context from first argument
    
    Returns:
        The decorated function with error handling.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        @with_error_handling(
            error_message=error_msg or f"Error in {func.__name__}",
            level=level,
            reraise=reraise,
            default=None
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

def handle_ui_errors(
    error_component_title: str = "UI Operation Error",
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = True,
    show_dialog: bool = True,
    create_ui: Optional[bool] = None,  # For backward compatibility
    return_type: Optional[type] = None,  # For backward compatibility
    log_error: bool = True,  # For backward compatibility
) -> Callable[[F], F]:
    """
    Decorator specifically for UI operations with user-friendly error handling.
    
    Args:
        error_component_title: Title for the error UI component
        level: Error severity level
        reraise: Whether to re-raise the exception
        show_dialog: Whether to show error dialog to user
        create_ui: Alias for show_dialog (for backward compatibility)
        return_type: For backward compatibility (ignored)
        log_error: For backward compatibility (ignored)
    
    Returns:
        Decorated function with UI error handling
    """
    # Use create_ui if explicitly provided, otherwise use show_dialog
    use_create_ui = show_dialog if create_ui is None else create_ui
    
    return handle_errors(
        error_msg=error_component_title,
        level=level,
        reraise=reraise,
        create_ui=use_create_ui,
        log_error=log_error
    )

def safe_ui_operation(
    operation_name: str = "ui_operation",
    level: ErrorLevel = ErrorLevel.ERROR,
    fallback_return: Any = None,
    component: str = "main_container"
) -> Callable[[F], F]:
    """
    Decorator for safely executing UI operations with fallback values.
    
    Args:
        operation_name: Name of the operation for error context
        level: Error severity level
        fallback_return: Value to return if an exception occurs
        component: Component name for error context
        
    Returns:
        Decorated function that safely handles exceptions
    """
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            @with_error_handling(
                error_message=f"Error in async {operation_name}",
                level=level,
                reraise=False,
                default=fallback_return,
                component=component,
                operation=operation_name
            )
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            @wraps(func)
            @with_error_handling(
                error_message=f"Error in {operation_name}",
                level=level,
                reraise=False,
                default=fallback_return,
                component=component,
                operation=operation_name
            )
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return cast(F, sync_wrapper)
    return decorator

def log_errors(
    level: Union[ErrorLevel, str] = ErrorLevel.ERROR,
    component: str = "main_container",
    operation: str = "unknown"
) -> Callable[[F], F]:
    """
    Decorator to log errors without interrupting execution.
    
    Args:
        level: The log level to use (can be string or ErrorLevel enum).
        component: The component where the error occurred.
        operation: The operation being performed.
        
    Returns:
        A decorator that logs errors and re-raises them.
    """
    # Convert string level to ErrorLevel if needed
    if isinstance(level, str):
        level = ErrorLevel[level.upper()]
    
    def decorator(func: F) -> F:
        @wraps(func)
        @with_error_handling(
            error_message=f"Error in {component}.{operation}",
            level=level,
            reraise=True,
            component=component,
            operation=operation
        )
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    return decorator

def suppress_errors() -> Callable[[F], F]:
    """
    Decorator to silently suppress all errors.
    
    Use with caution - only for cases where errors are truly non-critical.
    """
    return handle_errors(level=ErrorLevel.DEBUG, reraise=False, log_error=False)

def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts before giving up.
        delay: Initial delay between attempts in seconds.
        backoff: Multiplier for delay between attempts.
        exceptions: Tuple of exceptions to catch and retry on.
    """
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                current_delay = delay
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                
                # If we get here, all attempts failed
                raise last_exception  # type: ignore
            
            return cast(F, async_wrapper)
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                current_delay = delay
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts:
                            import time
                            time.sleep(current_delay)
                            current_delay *= backoff
                
                # If we get here, all attempts failed
                raise last_exception  # type: ignore
            
            return cast(F, sync_wrapper)
    return decorator

def safe_component_operation(
    error_message: Optional[Union[str, Callable]] = None,
    level: ErrorLevel = ErrorLevel.WARNING,
    default: Any = None
) -> Callable[[F], F]:
    """
    Decorator for safe component operations with default fallback.
    
    Args:
        error_message: Error message string or callable that generates message
        level: Error severity level
        default: Default value to return on error
        
    Returns:
        Decorated function with safe component handling
    """
    def decorator(func: F) -> F:
        @wraps(func)
        @with_error_handling(
            error_message=error_message or f"Error in component operation {func.__name__}",
            level=level,
            reraise=False,
            default=default,
            component="component_operation",
            operation=func.__name__
        )
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    return decorator
"""
Error handling decorators for SmartCash UI Core.

This module provides decorators for consistent error handling across the
application, allowing for clean and reusable error handling patterns.
"""
import functools
import logging
import inspect
import traceback
from functools import wraps
from typing import (
    Any, Callable, Dict, Optional, Type, TypeVar, cast, 
    Union, Awaitable, TypeVar, overload, Tuple, List
)

from .handlers import CoreErrorHandler, get_error_handler
from .context import ErrorContext
from .enums import ErrorLevel
from smartcash.common.exceptions import ErrorContext as CommonErrorContext

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# For backward compatibility with error_utils.py
UILogger = Any  # Type alias for UILogger


def handle_errors(
    error_msg: Optional[str] = None,
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = True,
    log_error: bool = True,
    create_ui: bool = False,
    handler: Optional[CoreErrorHandler] = None,
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
        handler: Optional error handler instance to use. If None, the default
                handler will be used.
    
    Returns:
        A decorator that applies the specified error handling.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the error handler instance
            error_handler = handler or get_error_handler()
            
            # Determine the error message to use
            msg = error_msg or getattr(func, '__doc__', 
                                     f"Error in {func.__name__}")
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Format the error message with function name and arguments
                formatted_msg = f"{msg}: {str(e)}"
                
                # Get the error context from the instance if context_attr is provided
                error_context = None
                if args and hasattr(args[0], context_attr):
                    error_context = getattr(args[0], context_attr)
                
                # Handle the error using the error handler
                result = error_handler.handle_error(
                    error_msg=formatted_msg,
                    level=level,
                    exc_info=True,
                    fail_fast=reraise,
                    create_ui_error=create_ui,
                    function_name=func.__name__,
                    error_type=type(e).__name__,
                    error_context=error_context,
                    **kwargs
                )
                
                # If not re-raising, return the result from the error handler
                if not reraise:
                    return result
                
                # Re-raise the original exception
                raise
        
        return cast(F, wrapper)
    return decorator


def log_errors(
    level: Union[ErrorLevel, str] = ErrorLevel.ERROR,
    component: str = "ui",
    operation: str = "unknown"
) -> Callable[[F], F]:
    """
    Decorator to log errors without interrupting execution.
    
    Args:
        level: The log level to use (can be string or ErrorLevel enum).
        component: The component where the error occurred.
        operation: The operation being performed.
        
    Returns:
        A decorator that logs errors without re-raising them.
    """
    # Convert string level to ErrorLevel if needed
    if isinstance(level, str):
        level = ErrorLevel[level.upper()]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the error handler
                handler = get_error_handler()
                
                # Create error context
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    details={
                        'function': func.__name__,
                        'error_type': type(e).__name__,
                        'message': str(e)
                    }
                )
                
                # Log the error
                handler.handle_error(
                    error_msg=f"Error in {component}.{operation}: {str(e)}",
                    level=level,
                    exc_info=True,
                    **asdict(context)
                )
                
                # Re-raise the exception
                raise
                
        # Handle async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Get the error handler
                    handler = get_error_handler()
                    
                    # Create error context
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        details={
                            'function': func.__name__,
                            'error_type': type(e).__name__,
                            'message': str(e)
                        }
                    )
                    
                    # Log the error
                    handler.handle_error(
                        error_msg=f"Error in {component}.{operation}: {str(e)}",
                        level=level,
                        exc_info=True,
                        **asdict(context)
                    )
                    
                    # Re-raise the exception
                    raise
                    
            return cast(F, async_wrapper)
            
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
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise
                    
                    # Log the retry attempt
                    get_error_handler().handle_error(
                        f"Attempt {attempt} failed, retrying in {current_delay:.1f}s: {e}",
                        level=ErrorLevel.WARNING,
                        exc_info=True,
                        fail_fast=False,
                        attempt=attempt,
                        max_attempts=max_attempts
                    )
                    
                    # Wait before retrying
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached due to the raise above
            raise last_exception  # type: ignore
        
        return cast(F, wrapper)
    return decorator


def safe_ui_operation(
    component: str = "ui", 
    operation: str = "unknown"
) -> Callable[[F], F]:
    """
    Decorator untuk menjalankan operasi UI dengan error handling yang aman.
    
    Args:
        component: Nama komponen yang menggunakan decorator
        operation: Nama operasi yang di-wrap
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the error handler
                handler = get_error_handler()
                
                # Format error message
                error_msg = f"Gagal {operation}: {str(e)}"
                
                # Create error context
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    details={
                        'function': func.__name__,
                        'error_type': type(e).__name__,
                        'message': str(e)
                    }
                )
                
                # Handle the error with UI components
                handler.handle_error(
                    error_msg=error_msg,
                    level=ErrorLevel.ERROR,
                    exc_info=True,
                    create_ui=True,
                    **asdict(context)
                )
                
                # Re-raise the exception
                raise
                
        # Handle async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Get the error handler
                    handler = get_error_handler()
                    
                    # Format error message
                    error_msg = f"Gagal {operation}: {str(e)}"
                    
                    # Create error context
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        details={
                            'function': func.__name__,
                            'error_type': type(e).__name__,
                            'message': str(e)
                        }
                    )
                    
                    # Handle the error with UI components
                    handler.handle_error(
                        error_msg=error_msg,
                        level=ErrorLevel.ERROR,
                        exc_info=True,
                        create_ui=True,
                        **asdict(context)
                    )
                    
                    # Re-raise the exception
                    raise
                    
            return cast(F, async_wrapper)
            
        return cast(F, wrapper)
    return decorator

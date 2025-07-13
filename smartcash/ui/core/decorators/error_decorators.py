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

from smartcash.ui.core.errors.handlers import CoreErrorHandler, get_error_handler
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
        context_attr: Attribute name to get error context from first argument
    
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
                if args and isinstance(context_attr, str) and context_attr.strip():
                    try:
                        if hasattr(args[0], context_attr):
                            error_context = getattr(args[0], context_attr, None)
                    except Exception as context_error:
                        # Log but don't fail if we can't get the context
                        error_handler._logger.warning(
                            f"Failed to get error context from {context_attr}: {str(context_error)}",
                            exc_info=True
                        )
                
                # Prepare error context
                error_kwargs = {
                    'level': level,
                    'exc_info': True,
                    'fail_fast': reraise,
                    'create_ui_error': create_ui,
                    'function_name': func.__name__,
                    'error_type': type(e).__name__,
                    'error_context': error_context
                }
                
                # Handle the error using the error handler
                result = error_handler.handle_error(
                    formatted_msg,
                    **error_kwargs
                )
                
                # If not re-raising, return the result from the error handler
                if not reraise:
                    return result
                
                # Re-raise the original exception
                raise
        
        return cast(F, wrapper)
    return decorator

def handle_ui_errors(
    error_component_title: str = "UI Operation Error",
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = True,
    show_dialog: bool = True
) -> Callable[[F], F]:
    """
    Decorator specifically for UI operations with user-friendly error handling.
    
    Args:
        error_component_title: Title for the error UI component
        level: Error severity level
        reraise: Whether to re-raise the exception
        show_dialog: Whether to show error dialog to user
    
    Returns:
        Decorated function with UI error handling
    """
    return handle_errors(
        error_msg=error_component_title,
        level=level,
        reraise=reraise,
        create_ui=show_dialog
    )

def safe_ui_operation(
    operation_name: str = "ui_operation",
    level: ErrorLevel = ErrorLevel.ERROR,
    fallback_return: Any = None,
    component: str = "ui"
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
                    operation=operation_name,
                    details={
                        'function': func.__name__,
                        'error_type': type(e).__name__,
                        'message': str(e)
                    }
                )
                
                # Handle the error with UI components
                handler.handle_error(
                    error_msg=f"Error in {component}.{operation_name}: {str(e)}",
                    level=level,
                    exc_info=True,
                    create_ui_error=True,
                    **asdict(context)
                )
                
                # Return fallback value instead of re-raising
                return fallback_return
                
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
                        operation=operation_name,
                        details={
                            'function': func.__name__,
                            'error_type': type(e).__name__,
                            'message': str(e)
                        }
                    )
                    
                    # Handle the error with UI components
                    handler.handle_error(
                        error_msg=f"Error in {component}.{operation_name}: {str(e)}",
                        level=level,
                        exc_info=True,
                        create_ui_error=True,
                        **asdict(context)
                    )
                    
                    # Return fallback value instead of re-raising
                    return fallback_return
                    
            return cast(F, async_wrapper)
            
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
        A decorator that logs errors and re-raises them.
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
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Generate error message
                if callable(error_message):
                    msg = error_message(e, *args, **kwargs)
                else:
                    msg = error_message or f"Component operation {func.__name__} failed: {e}"
                
                # Handle the error
                get_error_handler().handle_error(
                    error_msg=msg,
                    level=level,
                    exc_info=True,
                    fail_fast=False
                )
                
                return default
                
        return cast(F, wrapper)
    return decorator
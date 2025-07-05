"""
Utility functions for error handling in SmartCash UI Core.

This module provides utility functions for working with errors, exceptions,
and error handling in the SmartCash application.
"""
import functools
import inspect
import logging
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .enums import ErrorLevel
from .context import ErrorContext

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])


def get_exception_info(exception: BaseException) -> Dict[str, Any]:
    """
    Extract detailed information from an exception.
    
    Args:
        exception: The exception to extract information from.
        
    Returns:
        Dict containing exception information including type, message, and traceback.
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # Get the traceback as a list of strings
    tb_list = traceback.format_exception(
        type(exception), 
        exception, 
        exception.__traceback__
    ) if exc_traceback is not None else []
    
    # Get the last frame where the exception was raised
    tb = exception.__traceback__
    last_frame = None
    while tb is not None:
        last_frame = tb
        tb = tb.tb_next
    
    # Extract frame information if available
    frame_info = {}
    if last_frame is not None:
        frame = last_frame.tb_frame
        frame_info = {
            'filename': frame.f_code.co_filename,
            'lineno': frame.f_lineno,
            'function': frame.f_code.co_name,
            'locals': {
                k: str(v) for k, v in frame.f_locals.items()
                if not k.startswith('__') and not k.endswith('__')
            }
        }
    
    return {
        'type': type(exception).__name__,
        'message': str(exception),
        'module': exception.__class__.__module__,
        'traceback': ''.join(tb_list),
        'frame': frame_info,
        'attributes': {
            k: str(v) for k, v in vars(exception).items()
            if not k.startswith('__') and not k.endswith('__')
        }
    }


def format_error_message(
    message: str, 
    level: ErrorLevel = ErrorLevel.ERROR,
    **context: Any
) -> str:
    """
    Format an error message with context and styling.
    
    Args:
        message: The base error message.
        level: The severity level of the error.
        **context: Additional context to include in the formatted message.
        
    Returns:
        Formatted error message string.
    """
    # Add level prefix
    formatted = f"[{level.name}] {message}"
    
    # Add context if provided
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        formatted += f"\nContext: {context_str}"
    
    return formatted


def log_error(
    message: str,
    level: ErrorLevel = ErrorLevel.ERROR,
    logger: Optional[logging.Logger] = None,
    exc_info: bool = False,
    **context: Any
) -> None:
    """
    Log an error message with the specified level and context.
    
    Args:
        message: The error message to log.
        level: The severity level of the error.
        logger: Optional logger to use. If None, creates a new one.
        exc_info: Whether to include exception info in the log.
        **context: Additional context to include in the log.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Format the message with context
    formatted_msg = format_error_message(message, level, **context)
    
    # Log at the appropriate level
    if level == ErrorLevel.DEBUG:
        logger.debug(formatted_msg, exc_info=exc_info)
    elif level == ErrorLevel.INFO:
        logger.info(formatted_msg, exc_info=exc_info)
    elif level == ErrorLevel.WARNING:
        logger.warning(formatted_msg, exc_info=exc_info)
    elif level == ErrorLevel.ERROR:
        logger.error(formatted_msg, exc_info=exc_info)
    elif level == ErrorLevel.CRITICAL:
        logger.critical(formatted_msg, exc_info=exc_info)


def with_error_handling(
    func: Optional[F] = None,
    error_message: Optional[str] = None,
    level: ErrorLevel = ErrorLevel.ERROR,
    reraise: bool = True,
    default: Any = None
) -> Any:
    """
    Decorator to wrap a function with error handling.
    
    Args:
        func: The function to decorate.
        error_message: Custom error message to use.
        level: The severity level for logged errors.
        reraise: Whether to re-raise the exception after handling.
        default: Default value to return if an error occurs and reraise is False.
        
    Returns:
        The decorated function.
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                # Format the error message
                msg = error_message or f"Error in {f.__name__}: {str(e)}"
                
                # Log the error
                log_error(
                    msg,
                    level=level,
                    exc_info=True,
                    function=f.__name__,
                    args=args,
                    kwargs=kwargs
                )
                
                # Re-raise or return default value
                if reraise:
                    raise
                return default
        return cast(F, wrapper)
    
    # Handle both @with_error_handling and @with_error_handling() syntax
    if func is not None:
        return decorator(func)
    return decorator


def ignore_errors(
    func: Optional[F] = None,
    error_message: Optional[str] = None,
    level: ErrorLevel = ErrorLevel.DEBUG,
    default: Any = None
) -> Any:
    """
    Decorator to silently ignore errors in a function.
    
    Args:
        func: The function to decorate.
        error_message: Custom error message to use.
        level: The severity level for logged errors.
        default: Value to return if an error occurs.
        
    Returns:
        The decorated function.
    """
    return with_error_handling(
        func=func,
        error_message=error_message,
        level=level,
        reraise=False,
        default=default
    )


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = (Exception,)
    ) -> Callable[[F], F]:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts before giving up.
        delay: Initial delay between attempts in seconds.
        backoff: Multiplier for delay between attempts.
        exceptions: Tuple of exceptions to catch and retry on.
        
    Returns:
        A decorator that can be applied to functions.
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
                    log_error(
                        f"Attempt {attempt}/{max_attempts} failed, retrying in {current_delay:.1f}s",
                        level=ErrorLevel.WARNING,
                        exception=str(e),
                        function=func.__name__,
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


def get_caller_info(skip: int = 1) -> Dict[str, Any]:
    """
    Get information about the calling function.
    
    Args:
        skip: Number of frames to skip in the call stack.
        
    Returns:
        Dict containing information about the calling function.
    """
    frame = inspect.currentframe()
    for _ in range(skip + 1):
        if frame is None:
            break
        frame = frame.f_back
    
    if frame is None:
        return {}
    
    try:
        # Get the frame info
        frame_info = inspect.getframeinfo(frame)
        
        # Get the calling function name
        func_name = frame.f_code.co_name
        
        # Get the class name if this is a method
        class_name = None
        if 'self' in frame.f_locals:
            instance = frame.f_locals['self']
            class_name = instance.__class__.__name__
        
        return {
            'file': frame_info.filename,
            'line': frame_info.lineno,
            'function': func_name,
            'class': class_name,
            'module': frame.f_globals.get('__name__', 'unknown')
        }
    finally:
        # Avoid reference cycles
        del frame

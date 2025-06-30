"""
Standardized error handling utilities for pretrained services.

This module provides consistent error handling patterns for the pretrained services
module, ensuring all errors are properly logged and handled according to the project's
standards.
"""
from typing import Any, Callable, Dict, Optional, TypeVar, Type, Union, cast
from functools import wraps
import logging
import traceback

from smartcash.ui.utils.error_utils import (
    with_error_handling as base_with_error_handling,
    log_errors as base_log_errors,
    create_error_context,
    ErrorHandler
)
from smartcash.common.exceptions import SmartCashError, UIError

# Type variables for generic function typing
T = TypeVar('T')
LoggerType = Union[logging.Logger, Callable[[str, str], None]]

# Module logger
logger = logging.getLogger(__name__)

def get_logger(logger_bridge: Optional[LoggerType] = None) -> logging.Logger:
    """Get a logger instance, falling back to the module logger.
    
    Args:
        logger_bridge: Optional logger bridge or logger instance
        
    Returns:
        A logging.Logger instance
    """
    if logger_bridge is None:
        return logger
    if isinstance(logger_bridge, logging.Logger):
        return logger_bridge
    
    # For callable logger bridges, create an adapter
    class LoggerBridgeAdapter(logging.LoggerAdapter):
        def log(self, level, msg, *args, **kwargs):
            if self.isEnabledFor(level):
                self.logger(msg, logging.getLevelName(level).lower())
    
    return LoggerBridgeAdapter(logger, {})

def with_error_handling(
    component: str,
    operation: str,
    fallback_value: Any = None,
    fallback_factory: Optional[Callable[..., Any]] = None,
    log_level: str = "error"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for consistent error handling in pretrained services.
    
    This is a specialized version of error_utils.with_error_handling with
    defaults and conventions specific to pretrained services.
    
    Args:
        component: The component name (e.g., "pretrained", "model_checker")
        operation: The operation being performed (e.g., "get_model_info")
        fallback_value: Value to return on error if fallback_factory is not provided
        fallback_factory: Callable that returns a fallback value, called with
                         the same arguments as the decorated function
        log_level: Log level to use for errors
        
    Returns:
        A decorator that adds error handling to the function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create error context with component and operation
            ctx = create_error_context(
                component=component,
                operation=operation,
                details={
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
            )
            
            try:
                return func(*args, **kwargs)
            except SmartCashError as e:
                # Re-raise SmartCashErrors with additional context
                e.add_context(ctx)
                raise
            except Exception as e:
                # Log the error with full context
                logger.error(
                    f"Error in {component}.{operation}: {str(e)}",
                    exc_info=True,
                    extra={"context": ctx}
                )
                
                # Create and return fallback value
                if fallback_factory is not None:
                    try:
                        return fallback_factory(*args, **kwargs)
                    except Exception as factory_error:
                        logger.error(
                            f"Error in fallback_factory for {component}.{operation}: {str(factory_error)}",
                            exc_info=True
                        )
                        raise SmartCashError(
                            f"Error in fallback_factory: {str(factory_error)}",
                            context=ctx
                        ) from factory_error
                
                if fallback_value is not None:
                    return fallback_value
                
                # Re-raise with additional context
                raise SmartCashError(
                    f"Error in {component}.{operation}: {str(e)}",
                    context=ctx
                ) from e
        
        return wrapper
    return decorator

def log_errors(
    level: str = "error",
    logger_bridge: Optional[LoggerType] = None,
    **context: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for logging errors with context.
    
    This is a specialized version of error_utils.log_errors with additional
    context handling for pretrained services.
    
    Args:
        level: Log level ('error', 'warning', 'info', 'debug')
        logger_bridge: Optional logger bridge or logger instance
        **context: Additional context to include in logs
        
    Returns:
        A decorator that adds error logging to the function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            log = get_logger(logger_bridge)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create log record with context
                extra = {
                    "context": {
                        **context,
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                }
                
                # Log the error
                log_msg = f"Error in {func.__name__}: {str(e)}"
                if level == "error":
                    log.error(log_msg, exc_info=True, extra=extra)
                elif level == "warning":
                    log.warning(log_msg, exc_info=True, extra=extra)
                elif level == "info":
                    log.info(log_msg, exc_info=True, extra=extra)
                else:  # debug
                    log.debug(log_msg, exc_info=True, extra=extra)
                
                raise
        
        return wrapper
    return decorator

# Export public API
__all__ = [
    'with_error_handling',
    'log_errors',
    'get_logger',
]

"""
Core error handling functionality for SmartCash UI Core.

This module provides the main error handling classes and utilities for
consistent error handling across the application.
"""
import asyncio
import inspect
import logging
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, 
    Type, TypeVar, Union, cast, TypeVar, Awaitable, ContextManager, overload, Tuple
)

try:
    from IPython.display import display, HTML
except ImportError:
    # Fallback for environments without IPython
    def display(content):
        print(content)
    
    class HTML:
        def __init__(self, content):
            self.data = content
        
        def __str__(self):
            return self.data

# Removed circular import: from ..decorators.error_decorators import handle_ui_errors

from .enums import ErrorLevel
from .context import ErrorContext
from .error_component import create_error_component
from smartcash.common.exceptions import ErrorContext as CommonErrorContext

# Type variable for generic typing
T = TypeVar('T')

# Global error handler instance
_GLOBAL_ERROR_HANDLER = None


def get_error_handler() -> 'CoreErrorHandler':
    """
    Get the global error handler instance.
    
    If no global handler exists, creates and returns a default one.
    
    Returns:
        CoreErrorHandler: The global error handler instance.
    """
    global _GLOBAL_ERROR_HANDLER
    if _GLOBAL_ERROR_HANDLER is None:
        _GLOBAL_ERROR_HANDLER = CoreErrorHandler()
    return _GLOBAL_ERROR_HANDLER


def set_error_handler(handler: 'CoreErrorHandler') -> None:
    """
    Set the global error handler instance.
    
    Args:
        handler: The error handler instance to set as global.
    """
    global _GLOBAL_ERROR_HANDLER
    _GLOBAL_ERROR_HANDLER = handler


class CoreErrorHandler:
    """
    Centralized error handling for the SmartCash UI Core.
    
    This class provides methods for consistent error handling, logging,
    and user feedback across the application.
    """
    
    def __init__(
        self, 
        module_name: str = "SmartCash",
        ui_components: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        component_name: Optional[str] = None
    ):
        """
        Initialize the error handler.
        
        Args:
            module_name: The name of the module this handler is for.
            ui_components: Dictionary of UI components for error display.
            logger: Custom logger instance to use. If None, a default will be created.
            component_name: Optional name of the component this handler is for.
                           If provided, will be used as part of the module name.
        """
        self.module_name = component_name or module_name
        self._error_count = 0
        self._last_error = None
        self._ui_components = ui_components or {}
        self._logger = logger or self._get_logger()
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Optional logger name. If None, uses default module name.
            
        Returns:
            logging.Logger: The logger instance.
        """
        logger_name = name or f"smartcash.ui.core.errors.{self.module_name}"
        return self._get_logger_by_name(logger_name)
    
    def _get_logger_by_name(self, logger_name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            logger_name: Name for the logger
            
        Returns:
            logging.Logger: The logger instance.
        """
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            # Configure the logger if it hasn't been configured yet
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_logger(self) -> logging.Logger:
        """
        Get or create a logger for this error handler.
        
        Returns:
            logging.Logger: The logger instance.
        """
        return self._get_logger_by_name(f"smartcash.ui.core.errors.{self.module_name}")
    
    @property
    def error_count(self) -> int:
        """Get the total number of errors handled."""
        return self._error_count
    
    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message that was handled."""
        return self._last_error
    
    def reset_error_count(self) -> None:
        """Reset the error counter."""
        self._error_count = 0
        self._last_error = None
    
    def handle_error(
        self,
        error_msg: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        exc_info: bool = False,
        fail_fast: bool = False,
        create_ui_error: bool = False,
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Handle an error with the specified severity level.
        
        Args:
            error_msg: The error message to log/display.
            level: The severity level of the error.
            exc_info: Whether to include exception info in the log.
            fail_fast: If True, raises an exception for ERROR/CRITICAL levels.
            create_ui_error: If True, creates and returns a UI error component.
            **kwargs: Additional context to include in the error.
            
        Returns:
            Optional[Any]: A UI error component if create_ui_error is True,
                         otherwise None.
            
        Raises:
            RuntimeError: If fail_fast is True and level is ERROR/CRITICAL.
        """
        # Update error tracking
        self._error_count += 1
        self._last_error = error_msg
        
        # Get the calling frame information
        frame = inspect.currentframe().f_back
        caller_info = self._get_caller_info(frame) if frame else {}
        
        # Merge context from kwargs and caller info
        context = {**caller_info, **kwargs}
        
        # Log the error
        self._log_error(error_msg, level, exc_info, **context)
        
        # Create UI error component if requested
        ui_component = None
        if create_ui_error:
            ui_component = self._create_ui_error(error_msg, level, **context)
        
        # Raise an exception if this is a critical error and fail_fast is True
        if fail_fast and level in [ErrorLevel.ERROR, ErrorLevel.CRITICAL]:
            raise RuntimeError(f"[{self.module_name}] {error_msg}")
        
        return ui_component
    
    def handle_exception(
        self,
        error: Exception,
        error_msg: Optional[str] = None,
        level: ErrorLevel = ErrorLevel.ERROR,
        fail_fast: bool = False,
        create_ui_error: bool = False,
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Handle an exception with the specified severity level.
        
        Args:
            error: The exception to handle.
            error_msg: Custom error message. If None, uses str(error).
            level: The severity level of the error.
            fail_fast: If True, re-raises the exception after handling.
            create_ui_error: If True, creates and returns a UI error component.
            **kwargs: Additional context to include in the error.
            
        Returns:
            Optional[Any]: A UI error component if create_ui_error is True,
                         otherwise None.
        """
        error_msg = error_msg or str(error)
        
        # Get the traceback information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        has_exc_info = exc_type is not None and exc_value is not None
        
        # Format the error message with exception info if available
        if has_exc_info:
            tb_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
            error_msg = f"{error_msg}\n\n{' '.join(tb_list)}"
        
        # Include exception info in the context
        context = {
            'exception_type': error.__class__.__name__,
            'exception_msg': str(error),
            **kwargs
        }
        
        # Remove exc_info from context if it exists to avoid duplicate keyword argument
        context.pop('exc_info', None)
        
        # Handle the error
        return self.handle_error(
            error_msg=error_msg,
            level=level,
            exc_info=has_exc_info,
            fail_fast=fail_fast,
            create_ui_error=create_ui_error,
            **context
        )
    
    def _log_error(
        self,
        error_msg: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        exc_info: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Log an error message with the specified severity level.
        
        Args:
            error_msg: The error message to log.
            level: The severity level of the error.
            exc_info: Whether to include exception info in the log.
            **kwargs: Additional context to include in the log.
        """
        # Format the log message with context, safely handling any objects that might cause recursion
        context_str = ""
        if kwargs:
            try:
                # Safely convert each value to string, with recursion protection
                safe_kwargs = {}
                for k, v in kwargs.items():
                    try:
                        # Skip any values that might cause recursion
                        if isinstance(v, (int, float, str, bool, type(None))):
                            safe_kwargs[k] = str(v)
                        else:
                            safe_kwargs[k] = f"<{type(v).__name__}>"
                    except Exception:
                        safe_kwargs[k] = "<error_converting_value>"
                
                context_str = " | " + ", ".join(f"{k}={v}" for k, v in safe_kwargs.items())
            except Exception as e:
                context_str = f" | error_formatting_context: {str(e)[:100]}"
        
        log_msg = f"{error_msg}{context_str}"
        
        # Log the message at the appropriate level
        if level == ErrorLevel.DEBUG:
            self._logger.debug(log_msg, exc_info=exc_info)
        elif level == ErrorLevel.INFO:
            self._logger.info(log_msg, exc_info=exc_info)
        elif level == ErrorLevel.WARNING:
            self._logger.warning(log_msg, exc_info=exc_info)
        elif level == ErrorLevel.ERROR:
            self._logger.error(log_msg, exc_info=exc_info)
        elif level == ErrorLevel.CRITICAL:
            self._logger.critical(log_msg, exc_info=exc_info)
    
    def _create_ui_error(
        self,
        error_msg: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        ui_components: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Create a UI error component for display.
        
        Args:
            error_msg: The error message to display.
            level: The severity level of the error.
            ui_components: UI components to use for displaying the error.
            **kwargs: Additional context for the error display.
            
        Returns:
            Any: A UI component that can be displayed.
        """
        # Use provided UI components or fall back to instance components
        ui_components = ui_components or self._ui_components
        
        try:
            # Import from local error components
            from .error_component import create_error_component
            from .enums import ErrorLevel
            
            # Ensure level is an ErrorLevel enum value
            if not isinstance(level, ErrorLevel):
                level = ErrorLevel.ERROR
                
            # Get error type from kwargs or use level name
            error_type = kwargs.pop('error_type', level.name.lower())
            
            # Ensure we have a valid error type
            valid_types = ['error', 'warning', 'info', 'success']
            if error_type not in valid_types:
                error_type = 'error'
                
            # Extract traceback if available
            traceback = kwargs.pop('traceback', None)
            
            # Create and return an error display component
            return create_error_component(
                error_message=error_msg,
                error_type=error_type,
                traceback=traceback,
                **kwargs
            )
        except ImportError:
            # Fallback to a simple display if ErrorDisplay is not available
            color = level.color
            return HTML(f"""
                <div style="color: {color}; padding: 10px; border-left: 4px solid {color}; margin: 5px 0;">
                    <strong>{level.icon} {level.name.title()}:</strong> {error_msg}
                </div>
            """)
    
    def _get_caller_info(self, frame: Any) -> Dict[str, Any]:
        """
        Get information about the calling function.
        
        Args:
            frame: The frame object of the caller.
            
        Returns:
            Dict containing caller information.
        """
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
        except Exception:
            # If we can't get the caller info, return an empty dict
            return {}
    
    def __call__(
        self,
        error_msg: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        **kwargs: Any
    ) -> None:
        """
        Allow the handler to be called as a function for convenience.
        
        Args:
            error_msg: The error message to handle.
            level: The severity level of the error.
            **kwargs: Additional context for the error.
        """
        self.handle_error(error_msg, level=level, **kwargs)
        
    def wrap_async(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """
        Decorator for async functions to handle errors.
        
        Args:
            func: The async function to wrap.
            
        Returns:
            Wrapped async function with error handling.
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.handle_exception(e, fail_fast=True)
                raise
        return wrapper
        
    def wrap_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for sync functions to handle errors.
        
        Args:
            func: The sync function to wrap.
            
        Returns:
            Wrapped sync function with error handling.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle_exception(e, fail_fast=True)
                raise
        return wrapper
        
    def with_ui_components(self, ui_components: Dict[str, Any]) -> 'CoreErrorHandler':
        """
        Create a new error handler with the specified UI components.
        
        Args:
            ui_components: Dictionary of UI components for error display.
            
        Returns:
            A new CoreErrorHandler instance with the specified UI components.
        """
        return CoreErrorHandler(
            module_name=self.module_name,
            ui_components=ui_components,
            logger=self._logger
        )
        
    @contextmanager
    def error_handler_scope(
        self,
        component: str = "main_container",
        operation: str = "unknown",
        ui_components: Optional[Dict[str, Any]] = None,
        show_ui_error: bool = True,
        log_level: str = "error"
    ) -> Iterator[None]:
        """
        Context manager for scoped error handling.
        
        Args:
            component: The component name where the error occurred.
            operation: The operation being performed.
            ui_components: UI components for displaying errors.
            show_ui_error: Whether to show UI errors.
            log_level: Log level for errors.
            
        Yields:
            None
        """
        try:
            yield
        except Exception as e:
            error_context = ErrorContext(
                component=component,
                operation=operation,
                details={"log_level": log_level}
            )
            
            self.handle_exception(
                e,
                level=ErrorLevel[log_level.upper()],
                create_ui_error=show_ui_error,
                ui_components=ui_components or self._ui_components,
                **asdict(error_context)
            )
            raise


def create_error_response(
    error_message: str,
    error: Optional[Exception] = None,
    title: str = "Error",
    include_traceback: bool = True,
    return_type: type = dict,
    **kwargs
) -> Any:
    """
    Create an error response with proper formatting.
    
    Args:
        error_message: The error message
        error: Optional exception object
        title: Title for the error
        include_traceback: Whether to include traceback
        return_type: Type of response to return
        **kwargs: Additional context
    
    Returns:
        Error response of specified type
    """
    # Create error component for UI display
    traceback_str = None
    if error and include_traceback:
        traceback_str = traceback.format_exc()
    
    error_component = create_error_component(
        error_message=error_message,
        traceback=traceback_str,
        title=title,
        **kwargs
    )
    
    # Return appropriate response type
    if return_type == dict:
        return {
            'success': False,
            'error': error_message,
            'error_component': error_component,
            'traceback': traceback_str
        }
    elif return_type == bool:
        return False
    elif return_type == str:
        return error_message
    else:
        try:
            return return_type()
        except:
            return None


# Export for convenience
__all__ = [
    'CoreErrorHandler',
    'get_error_handler',
    'set_error_handler',
    'create_error_response',
    'ErrorLevel',
    'ErrorContext',
    'create_error_component',
]

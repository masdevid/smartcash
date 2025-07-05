"""
File: smartcash/ui/core/handlers/base_handler.py
Deskripsi: Base handler dengan fail-fast principle dan centralized error handling
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable, Type, TypeVar, Tuple

from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors import (
    SmartCashUIError,
    ErrorContext,
    ErrorLevel,
    handle_errors,
    safe_component_operation,
    get_error_handler
)

T = TypeVar('T')

class BaseHandler(ABC):
    """Base handler dengan fail-fast principle dan centralized error handling."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        """Initialize base handler.
        
        Args:
            module_name: Nama module (e.g., 'downloader')
            parent_module: Parent module (e.g., 'dataset')
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Ensure full_module_name is a string
        if not isinstance(self.full_module_name, str):
            self.full_module_name = str(self.full_module_name)
        
        # Initialize error handler and context
        self._error_handler = get_error_handler()
        self._error_context = ErrorContext(
            component=self.__class__.__name__,
            operation="__init__",
            details={
                'module_name': module_name,
                'parent_module': parent_module
            }
        )
        
        # Setup logger through error handler
        self.logger = self._error_handler.get_logger(f"smartcash.ui.{self.full_module_name}")
        
        # Internal state
        self._is_initialized = False
        self._error_count = 0
        self._last_error = None
        
        self.logger.debug(f"🚀 Initialized {self.__class__.__name__} for {self.full_module_name}")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    @property
    def error_count(self) -> int:
        return self._error_count
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
        
    @property
    def error_handler(self):
        """Get the error handler instance."""
        return self._error_handler
    
    from contextlib import contextmanager

    @contextmanager
    def error_context(self, context_msg: str, fail_fast: bool = True):
        """Context manager for contextualized error handling.

        Args:
            context_msg: Description of the operation being performed.
            fail_fast: If True, re-raise errors via ``handle_error``; otherwise just log.
        """
        try:
            yield
        except Exception as e:
            # Build full message
            full_msg = f"{context_msg} failed: {e}"
            if fail_fast:
                self.handle_error(full_msg, exc_info=True)
            else:
                # Only log and store error without raising
                self._error_count += 1
                self._last_error = full_msg
                self.logger.error(f"❌ {full_msg}", exc_info=True)
        finally:
            pass

    @contextmanager
    def execute_safely(self, func: Callable[..., T], *args, **kwargs) -> T:
        # Wrap the function call with safe_component_operation
        safe_func = safe_component_operation(
            self,  # component
            func.__name__,  # operation name
            component_name=self.__class__.__name__
        )(func)
        
        """Execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            SmartCashUIError: If the function raises an exception
        """
        try:
            # Update context for the operation
            self._error_context.operation = f"execute_safely:{func.__name__}"
            self._error_context.details.update({
                'function_name': func.__name__,
                'function_module': getattr(func, '__module__', 'unknown'),
                'args': str(args),
                'kwargs': str(kwargs)
            })
            
            return func(*args, **kwargs)
            
        except SmartCashUIError:
            # Re-raise our custom errors
            raise
            
        except Exception as e:
            error_msg = f"Error in {func.__name__}"
            self.handle_error(error_msg, exc_info=True, error_details=str(e))

    @handle_errors(context_attr='_error_context')
    def handle_error(self, error_msg: str = None, error: Exception = None, exc_info: bool = False, **kwargs) -> None:
        """Handle errors with proper logging and state management.
        
        Args:
            error_msg: Error message to log (alternative to error parameter)
            error: Exception object (alternative to error_msg)
            exc_info: If True, log exception info
            **kwargs: Additional context for the error
            
        Raises:
            SmartCashUIError: Always raises this error type for consistent error handling
        """
        # Support both error_msg and error parameters for backward compatibility
        if error is not None and error_msg is None:
            error_msg = str(error)
        elif error_msg is None:
            error_msg = "An unknown error occurred"
            
        self._error_count += 1
        self._last_error = error_msg
        
        # Update error context with any additional kwargs
        self._error_context.details.update(kwargs)
        
        # If we have an actual exception, include its details
        if error is not None:
            self._error_context.details['exception_type'] = error.__class__.__name__
            self._error_context.details['exception_args'] = str(error.args) if error.args else ''
        
        # Log error through error handler
        error_context = ErrorContext(
            component=self.__class__.__name__,
            operation="handle_error",
            details={
                'error_message': error_msg,
                **self._error_context.details
            }
        )
        
        if exc_info or error is not None:
            self._error_handler.handle_exception(
                error_msg,
                error_level='ERROR',
                context=error_context,
                exc_info=exc_info or error is not None
            )
        else:
            self._error_handler.log_error(
                error_msg,
                error_level='ERROR',
                context=error_context
            )
        
        # Always raise a SmartCashUIError for consistent error handling
        raise SmartCashUIError(
            f"Handler Error [{self.full_module_name}]: {error_msg}",
            context=error_context
        )
    
    def reset_error_state(self) -> None:
        """Reset error state."""
        self._error_count = 0
        self._last_error = None
        self.logger.debug(f"🔄 Reset error state for {self.full_module_name}")
    
    # Component management methods have been removed to eliminate dependency on UIComponentManager
    
    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """Initialize handler (to be implemented by subclasses)."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup handler resources."""
        self._component_manager.cleanup()
        self.logger.debug(f"🧹 Cleaned up {self.__class__.__name__}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if exc_type is not None:
            self.handle_error(f"Exception in context: {exc_val}", exc_info=True)
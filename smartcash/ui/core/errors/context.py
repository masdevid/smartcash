"""
Error context management for SmartCash UI Core.

This module provides context management for error handling, allowing for
contextual error information to be passed through the call stack.
"""
from typing import Any, Dict, Optional, TypeVar, Generic, Type
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading

from .enums import ErrorLevel

# Type variable for the context value
T = TypeVar('T')


class ErrorContext:
    """
    Manages error context in a thread-local manner.
    
    This class provides a way to attach contextual information to errors
    that can be accessed by error handlers and logging systems.
    """
    _local = threading.local()
    
    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """
        Get the current error context for the current thread.
        
        Returns:
            Dict[str, Any]: A dictionary containing the current error context.
        """
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return cls._local.context
    
    @classmethod
    def set_context(cls, **kwargs) -> None:
        """
        Set context values in the current thread's error context.
        
        Args:
            **kwargs: Key-value pairs to set in the context.
        """
        context = cls.get_context()
        context.update(kwargs)
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a value from the current thread's error context.
        
        Args:
            key: The key to look up in the context.
            default: The default value to return if the key is not found.
            
        Returns:
            The value associated with the key, or the default if not found.
        """
        return cls.get_context().get(key, default)
    
    @classmethod
    def clear(cls) -> None:
        """Clear the current thread's error context."""
        if hasattr(cls._local, 'context'):
            cls._local.context.clear()
    
    @classmethod
    @contextmanager
    def context(cls, **kwargs):
        """
        Context manager for setting error context.
        
        Example:
            with ErrorContext.context(user_id=123, request_id='abc'):
                # Code that might raise exceptions
                pass
        """
        old_context = cls.get_context().copy()
        cls.set_context(**kwargs)
        try:
            yield
        finally:
            cls._local.context = old_context
    
    @classmethod
    def get_error_context(cls, error: Exception) -> Dict[str, Any]:
        """
        Extract context from an exception object.
        
        Args:
            error: The exception to extract context from.
            
        Returns:
            Dict containing error context information.
        """
        context = {
            'error_type': error.__class__.__name__,
            'error_message': str(error),
            'module': getattr(error, '__module__', None) or 'unknown',
        }
        
        # Add any additional context from the error object
        if hasattr(error, '__dict__'):
            for key, value in error.__dict__.items():
                if not key.startswith('_'):
                    context[f'error_{key}'] = value
        
        return context

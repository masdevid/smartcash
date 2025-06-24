"""
File: smartcash/ui/utils/simple_logger.py
Simple logger implementation with console output and emoji support.
"""

import sys
import traceback
from typing import Any, Dict, Optional, Tuple, Type, Union


def create_simple_logger(name: str = 'SimpleLogger') -> 'SimpleLogger':
    """
    Create a simple console logger with emoji support.
    
    Args:
        name: Name for the logger
        
    Returns:
        An instance of SimpleLogger
    """
    return SimpleLogger(name)


class SimpleLogger:
    """Simple logger with emoji support for console output."""
    
    def __init__(self, name: str = 'SimpleLogger') -> None:
        """
        Initialize the simple logger.
        
        Args:
            name: Name for the logger
        """
        self.name = name
        self.level = 'INFO'
            
    def _log(self, level: str, msg: str, *args, **kwargs) -> None:
        """
        Centralized logging method.
        
        Args:
            level: Log level (debug, info, warning, error, critical, success)
            msg: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments (supports exc_info for exceptions)
        """
        prefix = {
            'debug': '🐛',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🔥',
            'success': '✅'
        }.get(level.lower(), '📝')
        
        # Handle exc_info if present
        exc_info = kwargs.pop('exc_info', None)
        if exc_info and exc_info != (None, None, None):
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
            
            # Print the message first
            print(f"{prefix} {msg}")
            # Then print the traceback
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], 
                                   file=sys.stderr)
        else:
            print(f"{prefix} {msg}")
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self._log('debug', msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message."""
        self._log('info', msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self._log('warning', msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message."""
        self._log('error', msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self._log('critical', msg, *args, **kwargs)
        
    def success(self, msg: str, *args, **kwargs) -> None:
        """Log a success message."""
        self._log('success', msg, *args, **kwargs)
        
    def exception(self, msg: str, *args, **kwargs) -> None:
        """
        Log an exception message with traceback.
        
        This is a convenience method for logging exceptions.
        """
        kwargs['exc_info'] = True
        self._log('error', msg, *args, **kwargs)

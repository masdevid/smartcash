"""
Centralized Log Suppression Decorators

This module consolidates all log suppression decorators used across SmartCash,
providing consistent log management during UI initialization.
"""

import logging
import time
from typing import Set, Dict, Any, Optional, Callable, TypeVar
from contextlib import contextmanager
from threading import Lock
from functools import wraps

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

# Global state for log suppression
_suppression_active = False
_suppression_start_time = None
_suppression_duration = 5.0  # Default 5 seconds
_suppressed_loggers: Set[str] = set()
_original_log_levels: Dict[str, int] = {}
_suppression_lock = Lock()

class LogSuppressionFilter(logging.Filter):
    """Filter to suppress logs during UI initialization."""
    
    def __init__(self, suppressed_namespaces: Optional[Set[str]] = None):
        """Initialize the log suppression filter.
        
        Args:
            suppressed_namespaces: Set of logger namespaces to suppress
        """
        super().__init__()
        self.suppressed_namespaces = suppressed_namespaces or set()
        
    def filter(self, record):
        """Filter log records based on suppression state.
        
        Args:
            record: LogRecord to filter
            
        Returns:
            False if log should be suppressed, True otherwise
        """
        global _suppression_active, _suppression_start_time, _suppression_duration
        
        # Check if suppression is active and within time window
        if _suppression_active and _suppression_start_time:
            elapsed = time.time() - _suppression_start_time
            if elapsed <= _suppression_duration:
                # Check if this logger should be suppressed
                logger_name = record.name
                for namespace in self.suppressed_namespaces:
                    if logger_name.startswith(namespace):
                        return False
                        
        return True

def activate_log_suppression(
    duration: float = 5.0,
    suppressed_namespaces: Optional[Set[str]] = None
):
    """Activate log suppression for UI initialization.
    
    Args:
        duration: Duration in seconds to suppress logs
        suppressed_namespaces: Set of logger namespaces to suppress
    """
    global _suppression_active, _suppression_start_time, _suppression_duration
    global _suppressed_loggers, _original_log_levels
    
    with _suppression_lock:
        if _suppression_active:
            return  # Already active
            
        _suppression_active = True
        _suppression_start_time = time.time()
        _suppression_duration = duration
        
        # Default namespaces to suppress
        if suppressed_namespaces is None:
            suppressed_namespaces = {
                'smartcash.ui',
                'smartcash.dataset',
                'smartcash.common',
                'IPython',
                'ipywidgets',
                'traitlets'
            }
        
        # Create and add filter to root logger
        filter_obj = LogSuppressionFilter(suppressed_namespaces)
        
        # Apply filter to commonly used loggers
        loggers_to_filter = [
            '',  # Root logger
            'smartcash',
            'smartcash.ui',
            'smartcash.dataset',
            'smartcash.common'
        ]
        
        for logger_name in loggers_to_filter:
            logger = logging.getLogger(logger_name)
            logger.addFilter(filter_obj)
            _suppressed_loggers.add(logger_name)

def deactivate_log_suppression():
    """Deactivate log suppression and restore normal logging."""
    global _suppression_active, _suppression_start_time
    global _suppressed_loggers, _original_log_levels
    
    with _suppression_lock:
        if not _suppression_active:
            return  # Already inactive
            
        _suppression_active = False
        _suppression_start_time = None
        
        # Remove filters from all suppressed loggers
        for logger_name in _suppressed_loggers:
            logger = logging.getLogger(logger_name)
            # Remove all LogSuppressionFilter instances
            logger.filters = [
                f for f in logger.filters 
                if not isinstance(f, LogSuppressionFilter)
            ]
        
        _suppressed_loggers.clear()
        _original_log_levels.clear()

@contextmanager
def suppress_initial_logs(
    duration: float = 5.0,
    suppressed_namespaces: Optional[Set[str]] = None
):
    """Context manager to temporarily suppress logs during UI initialization.
    
    Args:
        duration: Duration in seconds to suppress logs
        suppressed_namespaces: Set of logger namespaces to suppress
        
    Example:
        with suppress_initial_logs(duration=3.0):
            # UI initialization code here
            create_ui_components()
    """
    activate_log_suppression(duration, suppressed_namespaces)
    try:
        yield
    finally:
        # Schedule deactivation after duration
        import threading
        def delayed_deactivation():
            time.sleep(duration)
            deactivate_log_suppression()
        
        thread = threading.Thread(target=delayed_deactivation, daemon=True)
        thread.start()

def is_suppression_active() -> bool:
    """Check if log suppression is currently active.
    
    Returns:
        True if suppression is active, False otherwise
    """
    global _suppression_active, _suppression_start_time, _suppression_duration
    
    if not _suppression_active:
        return False
        
    if _suppression_start_time:
        elapsed = time.time() - _suppression_start_time
        if elapsed > _suppression_duration:
            deactivate_log_suppression()
            return False
            
    return True

def extend_suppression(additional_duration: float):
    """Extend the current suppression duration.
    
    Args:
        additional_duration: Additional time in seconds to extend suppression
    """
    global _suppression_duration
    
    with _suppression_lock:
        if _suppression_active:
            _suppression_duration += additional_duration

class SuppressInitialLogs:
    """Decorator class to suppress logs during function execution."""
    
    def __init__(self, duration: float = 3.0, suppressed_namespaces: Optional[Set[str]] = None):
        """Initialize the decorator.
        
        Args:
            duration: Duration in seconds to suppress logs
            suppressed_namespaces: Set of logger namespaces to suppress
        """
        self.duration = duration
        self.suppressed_namespaces = suppressed_namespaces
        
    def __call__(self, func):
        """Decorate a function to suppress logs during execution.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with suppress_initial_logs(self.duration, self.suppressed_namespaces):
                return func(*args, **kwargs)
        
        # Set the __wrapped__ attribute for introspection
        wrapper.__wrapped__ = func
        wrapper.__name__ = getattr(func, '__name__', 'wrapped')
        wrapper.__doc__ = getattr(func, '__doc__', None)
        
        return wrapper

# Convenience decorators for common use cases
def suppress_ui_init_logs(duration: float = 3.0) -> Callable[[F], F]:
    """Decorator to suppress UI initialization logs.
    
    Args:
        duration: Duration in seconds to suppress logs
    """
    return SuppressInitialLogs(
        duration=duration,
        suppressed_namespaces={
            'smartcash.ui',
            'IPython',
            'ipywidgets',
            'traitlets'
        }
    )

def suppress_all_init_logs(duration: float = 5.0) -> Callable[[F], F]:
    """Decorator to suppress all SmartCash initialization logs.
    
    Args:
        duration: Duration in seconds to suppress logs
    """
    return SuppressInitialLogs(
        duration=duration,
        suppressed_namespaces={
            'smartcash',
            'IPython',
            'ipywidgets',
            'traitlets'
        }
    )
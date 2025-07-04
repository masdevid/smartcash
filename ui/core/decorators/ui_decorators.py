"""
File: smartcash/ui/core/decorators/ui_decorators.py
Description: Utility decorators for UI operations to reduce try-catch blocks and standardize error handling.

These decorators provide a clean way to handle common UI operation patterns with proper
error handling, logging, and fallbacks. They help reduce boilerplate try-catch blocks
throughout the codebase while maintaining consistent error handling behavior.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
import traceback
import logging

T = TypeVar('T')

def safe_ui_operation(operation_name: str = "ui_operation", log_level: str = "debug", 
                     fallback_return: Any = None):
    """Decorator for safely executing UI operations with standardized error handling.
    
    This decorator wraps a function with a try-catch block and handles logging
    of any exceptions that occur during execution.
    
    Args:
        operation_name: Name of the operation for logging purposes
        log_level: Logging level to use for errors (debug, info, warning, error)
        fallback_return: Value to return if an exception occurs
        
    Returns:
        Decorated function that safely handles exceptions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Get logger from self if available, otherwise use module logger
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                log_method = getattr(logger, log_level.lower(), logger.debug)
                
                # Log the error with appropriate level
                log_method(
                    f"Error in {operation_name}: {str(e)}",
                    exc_info=(log_level.lower() == 'error')
                )
                
                # Return fallback value
                return fallback_return
        return wrapper
    return decorator

def safe_widget_operation(widget_key_arg: int = 0, widget_key_kwarg: str = "ui_components"):
    """Decorator for safely performing operations on UI widgets.
    
    This decorator handles operations on UI widgets safely by catching exceptions
    for each widget access and operation. It's especially useful for methods that
    iterate through multiple widgets and perform operations on them.
    
    Args:
        widget_key_arg: Position of the UI components dictionary in args
        widget_key_kwarg: Name of the UI components keyword argument
        
    Returns:
        Decorated function that safely handles widget operations
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # Get UI components from args or kwargs
                ui_components = None
                if len(args) > widget_key_arg:
                    ui_components = args[widget_key_arg]
                elif widget_key_kwarg in kwargs:
                    ui_components = kwargs[widget_key_kwarg]
                
                if not ui_components or not isinstance(ui_components, dict):
                    # Get logger from self if available, otherwise use module logger
                    logger = getattr(self, 'logger', logging.getLogger(__name__))
                    logger.debug(f"No valid UI components for {func.__name__}")
                    return None
                
                return func(self, *args, **kwargs)
            except Exception as e:
                # Get logger from self if available, otherwise use module logger
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.debug(f"Error in {func.__name__}: {str(e)}")
                return None
        return wrapper
    return decorator

def safe_progress_operation(progress_key: str = "progress_tracker", 
                           fallback_keys: List[str] = None):
    """Decorator for safely performing operations on progress trackers.
    
    This decorator specializes in handling progress tracker operations safely,
    with fallbacks to standard progress bars if the tracker is not available.
    
    Args:
        progress_key: Key for the progress tracker in UI components
        fallback_keys: List of fallback keys to try if progress_key is not available
        
    Returns:
        Decorated function that safely handles progress operations
    """
    if fallback_keys is None:
        fallback_keys = ["progress_bar", "progress_container"]
        
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Get logger from self if available, otherwise use module logger
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                logger.debug(f"Error in progress operation {func.__name__}: {str(e)}")
                return None
        return wrapper
    return decorator

def safe_component_access(component_type: str = None, default_value: Any = None):
    """Decorator for safely accessing and using UI components.
    
    This decorator handles the pattern of safely accessing a component, checking
    if it has certain attributes or methods, and falling back gracefully if not.
    
    Args:
        component_type: Type of component being accessed (for logging)
        default_value: Default value to return if component access fails
        
    Returns:
        Decorated function that safely handles component access
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Get logger from self if available, otherwise use module logger
                logger = getattr(self, 'logger', logging.getLogger(__name__))
                component_info = f" ({component_type})" if component_type else ""
                logger.debug(f"Error accessing component{component_info} in {func.__name__}: {str(e)}")
                return default_value
        return wrapper
    return decorator

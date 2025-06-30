"""
File: smartcash/ui/utils/fallback_utils.py

Essential fallback utilities for UI error handling.

This module provides minimal, reliable error handling components that work even
when other parts of the system are not available. It uses the error_component.py
for rendering errors and fails fast on any issues.
"""

from typing import Any, Dict, Optional, Tuple, TypeVar, Callable, Type
import traceback
from dataclasses import dataclass

from smartcash.ui.components.error.error_component import create_error_component
from smartcash.ui.utils.ui_logger import get_module_logger

# Type variable for generic function returns
T = TypeVar('T')

# Get module logger
logger = get_module_logger(__name__)

@dataclass
class FallbackConfig:
    """Configuration for fallback UI components"""
    title: str = "⚠️ Error"
    message: str = "An error occurred"
    traceback: str = ""
    module_name: str = ""
    show_traceback: bool = True
    error_type: str = "error"

def create_fallback_ui(
    error_message: str,
    module_name: str = "module",
    ui_components: Optional[Dict[str, Any]] = None,
    exc_info: Optional[Tuple[Type[BaseException], BaseException, Any]] = None,
    config: Optional[FallbackConfig] = None
) -> Dict[str, Any]:
    """Create a fallback UI using ErrorComponent.
    
    Args:
        error_message: The error message to display
        module_name: Name of the module where the error occurred
        ui_components: Optional dictionary of UI components
        exc_info: Optional exception info tuple (type, value, traceback)
        config: Optional FallbackConfig with additional settings
        
    Returns:
        Dictionary containing the error UI and related components
    """
    try:
        # Initialize config with defaults if not provided
        if config is None:
            config = FallbackConfig()
        
        # Get traceback if not provided but exception info is available
        tb_text = config.traceback
        if exc_info and not tb_text:
            try:
                if exc_info[0] is not None and exc_info[1] is not None:
                    tb_text = ''.join(traceback.format_exception(*exc_info))
            except Exception:
                tb_text = "Failed to generate traceback"
        
        # Create error component
        error_ui = create_error_component(
            error_message=str(error_message or config.message),
            traceback=tb_text if config.show_traceback else None,
            title=str(config.title or f"Error in {module_name or 'module'}"),
            error_type=config.error_type or "error"
        )
        
        # Prepare return value
        result = {
            'ui': error_ui['widget'],
            'error': str(error_message or config.message),
            'fallback_mode': True,
            'error_details': {
                'module': str(module_name or 'unknown'),
                'message': str(error_message or config.message),
                'traceback': tb_text if config.show_traceback and tb_text else None
            }
        }
        
        return result
        
    except Exception as e:
        # If we can't create the error UI, raise immediately
        logger.error("Failed to create fallback UI", exc_info=True)
        raise

def safe_execute(
    func: Callable[..., T],
    error_message: str = "",
    default: Any = None,
    **kwargs
) -> T:
    """Safely execute a function and handle any exceptions.
    
    Args:
        func: The function to execute
        error_message: Custom error message to use if execution fails
        default: Default value to return on failure
        **kwargs: Arguments to pass to the function
        
    Returns:
        The function's return value or the default value on error
    """
    try:
        return func(**kwargs)
    except Exception as e:
        msg = error_message or f"Error in {getattr(func, '__name__', 'unknown')}"
        logger.error(f"{msg}: {str(e)}", exc_info=True)
        return default

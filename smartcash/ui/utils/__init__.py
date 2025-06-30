"""
File: smartcash/ui/utils/__init__.py
Deskripsi: Package untuk UI utilities
"""

from .widget_utils import display_widget, safe_display
from .error_handler import ErrorHandler
from .error_utils import (
    create_error_context,
    error_handler_scope,
    with_error_handling,
    log_errors
)
from .ui_logger import (
    UILogger,
    get_module_logger,
    setup_global_logging,
    log_to_ui,
    get_current_ui_logger,
    LoggerType
)

# For backward compatibility
LoggerBridge = UILogger
get_logger = get_module_logger  # Alias for backward compatibility

__all__ = [
    # Core utilities
    'display_widget',
    'safe_display',
    'ErrorHandler',
    
    # Error handling
    'create_error_context',
    'error_handler_scope',
    'with_error_handling',
    'log_errors',
    
    # Logging
    'UILogger',
    'get_module_logger',
    'get_logger',  # Backward compatibility
    'setup_global_logging',
    'log_to_ui',
    'get_current_ui_logger',
    'LoggerType',
    'LoggerBridge'  # Backward compatibility
]

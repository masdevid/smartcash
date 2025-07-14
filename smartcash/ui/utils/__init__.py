"""
File: smartcash/ui/utils/__init__.py
Deskripsi: Package untuk UI utilities

Modul ini menyediakan utilitas umum untuk komponen UI, termasuk:
- Manajemen widget
- Penanganan error terpusat
- Sistem logging terpadu
- Fungsi utilitas umum
"""

from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable, Tuple

# Import core utilities
from .widget_utils import display_widget, safe_display
from .display_utils import safe_display as safe_display_zmq

# Error handling utilities have been moved to smartcash.ui.core.errors
# Importing them here would cause circular imports
# Please import them directly from smartcash.ui.core.errors instead

# Import logging utilities
from smartcash.ui.logger import (
    UILogger,
    get_module_logger,
    get_ui_logger as get_current_ui_logger,
    LogLevel as LoggerType
)

# Backward compatibility
def log_to_ui(message: str, level: str = 'info', **kwargs):
    """
    Log a message to the UI logger.
    
    Args:
        message: The message to log
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        **kwargs: Additional arguments to pass to the logger
    """
    logger = get_current_ui_logger()
    log_method = getattr(logger, level, logger.info)
    log_method(message, **kwargs)

def setup_global_logging(level: str = 'info', **kwargs):
    """
    Set up global logging configuration.
    
    Args:
        level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        **kwargs: Additional arguments to pass to the logger
    """
    # This is now a no-op as logging is configured per-logger
    pass

# For backward compatibility
LoggerBridge = UILogger
get_logger = get_module_logger  # Alias for backward compatibility

# Re-export commonly used types for better IDE support
__all__ = [
    # Core utilities
    'display_widget',
    'safe_display',
    'safe_display_zmq',
    
    # Logging
    'UILogger',
    'LoggerBridge',
    'get_module_logger',
    'get_logger',
    'setup_global_logging',
    'log_to_ui',
    'get_current_ui_logger',
    'LoggerType',
    
    # Common types for type hints
    'Path',
    'Optional',
    'Any',
    'Dict',
    'List',
    'Union',
    'Callable',
    'Tuple'
]

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

# Import error handling utilities
from .error_utils import (
    create_error_context,
    error_handler_scope,
    with_error_handling,
    log_errors,
    ErrorHandler
)

# Import logging utilities
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

# Re-export commonly used types for better IDE support
__all__ = [
    # Core utilities
    'display_widget',
    'safe_display',
    
    # Error handling
    'ErrorHandler',
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
    'LoggerBridge',  # Backward compatibility
    
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

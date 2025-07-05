"""
File: smartcash/ui/__init__.py
Deskripsi: UI package exports and logger interface
"""

# Import subpackages to make them available as smartcash.ui.*
from .logger import UILogger, get_ui_logger, get_logger, get_module_logger, LogLevel

# Ensure all packages are properly imported
try:
    # Core UI components
    from . import core

    # UI Components
    from . import components

    # Setup and other components
    from . import types
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import UI modules: {e}")

# Explicitly list all exported modules and symbols
__all__ = [
    # Core modules
    'core',
    'components',
    'types',
    
    # Logger interface
    'UILogger',
    'get_ui_logger',
    'get_logger',
    'get_module_logger',
    'LogLevel',
]

# Create a default logger instance for the UI package
logger = get_ui_logger('smartcash.ui')

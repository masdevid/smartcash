"""
File: smartcash/ui/__init__.py
Deskripsi: UI package exports and logger interface
"""

import sys
import importlib
from typing import Dict, Any

# Import logger first to ensure it's available for error handling
from .logger import UILogger, get_ui_logger, get_logger, get_module_logger, LogLevel

# Initialize a logger for this module
_logger = get_module_logger(__name__)

# Track which modules failed to import
_failed_imports: Dict[str, str] = {}

def _import_module(module_name: str) -> None:
    """Helper function to import a module with proper error handling."""
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = module
    except ImportError as e:
        _failed_imports[module_name] = str(e)
        _logger.warning(f"Could not import UI module '{module_name}': {str(e)}")
        _logger.debug(f"Detailed import error for {module_name}:", exc_info=True)
    except Exception as e:
        _failed_imports[module_name] = f"Unexpected error: {str(e)}"
        _logger.error("Unexpected error importing UI module '%s'", module_name)
        _logger.debug("Detailed error:", exc_info=True)

# Import all required modules
for _module in ['core', 'components', 'types']:
    _import_module(_module)

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

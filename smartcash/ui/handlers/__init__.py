"""
File: smartcash/ui/handlers/__init__.py

⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.
Please update your imports to use smartcash.ui.core.handlers instead.
"""
import warnings

# Issue deprecation warning
warnings.warn(
    "The 'smartcash.ui.handlers' module is deprecated and will be removed in a future version. "
    "Please update your imports to use 'smartcash.ui.core.handlers' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import core handlers
from smartcash.ui.core.handlers import (
    BaseHandler,
    ConfigHandler as CoreConfigHandler,
    ConfigurableHandler,
    PersistentConfigHandler,
    SharedConfigHandler,
    UIHandler,
    ModuleUIHandler,
    OperationHandler
)

# Import error handler utilities
from smartcash.ui.core.errors.handlers import (
    handle_ui_errors,
    create_error_response,
    ErrorHandler
)

# For backward compatibility
ConfigHandler = CoreConfigHandler
DataHandler = CoreConfigHandler  # Alias for backward compatibility

__all__ = [
    # Core handlers
    'BaseHandler',
    'ConfigHandler',
    'ConfigurableHandler',
    'PersistentConfigHandler',
    'SharedConfigHandler',
    'UIHandler',
    'ModuleUIHandler',
    'OperationHandler',
    
    # Error handling
    'handle_ui_errors',
    'create_error_response',
    'ErrorHandler',
    
    # Backward compatibility
    'DataHandler'  # Deprecated
]
"""
File: smartcash/ui/core/shared/__init__.py
Deskripsi: Export shared utilities untuk SmartCash UI core module
"""

# Import directly from modules to avoid circular imports
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.core.errors.context import ErrorContext
from smartcash.ui.core.errors.handlers import CoreErrorHandler, get_error_handler
from smartcash.ui.core.decorators.error_decorators import (
    handle_errors,
    safe_component_operation
)
from smartcash.ui.core.errors.validators import (
    handle_component_validation,
    validate_ui_components
)
from smartcash.ui.core.shared.shared_config_manager import (
    SharedConfigManager,
    get_shared_config_manager
)

# Public exports
__all__ = [
    # Error handler
    'CoreErrorHandler',
    'ErrorLevel',
    'ErrorContext',
    'get_error_handler',
    'handle_component_validation',
    'safe_component_operation',
    'validate_ui_components',
    'handle_errors',
    
    # Config manager
    'SharedConfigManager',
    'get_shared_config_manager',
]
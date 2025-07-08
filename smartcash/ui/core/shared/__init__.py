"""
File: smartcash/ui/core/shared/__init__.py
Deskripsi: Export shared utilities untuk SmartCash UI core module
"""

# Import shared components
from smartcash.ui.core.errors import (
    CoreErrorHandler,
    ErrorLevel,
    ErrorContext,
    get_error_handler,
    handle_component_validation,
    safe_component_operation,
    validate_ui_components,
    handle_errors
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
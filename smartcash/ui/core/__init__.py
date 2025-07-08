"""
File: smartcash/ui/core/__init__.py
Deskripsi: Core UI module dengan fail-fast principle
"""

# Handlers
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.handlers.ui_handler import UIHandler

# Initializers
from smartcash.ui.core.initializers.base_initializer import BaseInitializer

# Shared utilities
from smartcash.ui.core.shared.shared_config_manager import SharedConfigManager, get_shared_config_manager

# Error handling
from smartcash.ui.core.errors import (
    ErrorLevel,
    CoreErrorHandler,
    get_error_handler,
    handle_errors,
    ErrorContext,
    handle_component_validation,
    safe_component_operation,
    validate_ui_components,
    ErrorComponent,
    create_error_component
)

__all__ = [
    'BaseHandler',
    'UIHandler',
    'BaseInitializer',
    'SharedConfigManager',
    'get_shared_config_manager',
    'ErrorLevel',
    'CoreErrorHandler',
    'get_error_handler',
    'handle_errors',
    'ErrorContext',
    'handle_component_validation',
    'safe_component_operation',
    'validate_ui_components',
    'ErrorComponent',
    'create_error_component'
]
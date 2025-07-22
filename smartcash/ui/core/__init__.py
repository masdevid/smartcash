"""
File: smartcash/ui/core/__init__.py
Deskripsi: Core UI module dengan fail-fast principle
"""

# Handlers - removed (deprecated)
# from smartcash.ui.core.handlers.base_handler import BaseHandler
# from smartcash.ui.core.handlers.ui_handler import UIHandler

# Initializers - removed (deprecated)
# from smartcash.ui.core.initializers.base_initializer import BaseInitializer

# Error handling
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.core.errors.handlers import CoreErrorHandler, get_error_handler
from smartcash.ui.core.errors.context import ErrorContext
from smartcash.ui.core.errors.validators import handle_component_validation, validate_ui_components
from smartcash.ui.core.errors.error_component import ErrorComponent, create_error_component

# Centralized decorators (all UI decorators unified here)
from smartcash.ui.core.decorators import (
    # Error handling decorators
    handle_errors,
    handle_ui_errors,
    safe_ui_operation,
    log_errors,
    suppress_errors,
    retry_on_failure,
    safe_component_operation,
    
    # Log suppression decorators
    suppress_ui_init_logs,
    suppress_all_init_logs,
    suppress_initial_logs,
    
    # UI operation decorators
    safe_widget_operation,
    safe_progress_operation,
    safe_component_access,
    safe_button_operation,
    safe_form_operation
)

__all__ = [
    # Handlers - removed (deprecated)
    # 'BaseHandler',
    # 'UIHandler',
    # 'BaseInitializer',
    
    # Error handling
    'ErrorLevel',
    'CoreErrorHandler',
    'get_error_handler',
    'ErrorContext',
    'handle_component_validation',
    'validate_ui_components',
    'ErrorComponent',
    'create_error_component',
    
    # Centralized decorators
    'handle_errors',
    'handle_ui_errors',
    'safe_ui_operation',
    'log_errors',
    'suppress_errors',
    'retry_on_failure',
    'safe_component_operation',
    'suppress_ui_init_logs',
    'suppress_all_init_logs',
    'suppress_initial_logs',
    'safe_widget_operation',
    'safe_progress_operation',
    'safe_component_access',
    'safe_button_operation',
    'safe_form_operation'
]
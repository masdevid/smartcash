"""
Error handling module for SmartCash UI Core.

This module provides a centralized error handling system with support for logging,
UI error components, and configurable error handling strategies.
"""
from .enums import ErrorLevel
from .handlers import CoreErrorHandler, get_error_handler
from .decorators import handle_errors
from .validators import (
    handle_component_validation,
    safe_component_operation,
    validate_ui_components
)
from .error_component import ErrorComponent, create_error_component
from .exceptions import (
    # Base exceptions
    SmartCashUIError,
    
    # UI exceptions
    UIError,
    UIComponentError,
    UIActionError,
    
    # Common exceptions
    ConfigError,
    ValidationError,
    NotSupportedError,
    
    # Error context
    ErrorContext
)

__all__ = [
    # Core error handling
    'ErrorLevel',
    'CoreErrorHandler',
    'get_error_handler',
    'handle_errors',
    'ErrorContext',
    'handle_component_validation',
    'safe_component_operation',
    'validate_ui_components',
    'ErrorComponent',
    'create_error_component',
    
    # Base exceptions
    'SmartCashUIError',
    
    # UI exceptions
    'UIError',
    'UIComponentError',
    'UIActionError',
    
    # Common exceptions
    'ConfigError',
    'ValidationError',
    'NotSupportedError'
]

"""
Error handling module for SmartCash UI Core.

This module provides a centralized error handling system with support for logging,
UI error components, and configurable error handling strategies.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .handlers import CoreErrorHandler  # noqa: F401

# Lazy imports to avoid circular dependencies
from .enums import ErrorLevel
from .context import ErrorContext
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
    NotSupportedError
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

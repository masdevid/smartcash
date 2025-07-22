"""
File: smartcash/ui/core/shared/__init__.py
Deskripsi: Export shared utilities untuk SmartCash UI core module
"""

# Basic exports to avoid circular imports
try:
    from smartcash.ui.core.errors.enums import ErrorLevel
    from smartcash.ui.core.errors.context import ErrorContext
except ImportError:
    # Handle import errors gracefully during initialization
    ErrorLevel = None
    ErrorContext = None

# Use lazy imports for complex error handlers that may cause circular imports
def get_error_handler():
    """Lazy import for error handler to avoid circular imports."""
    try:
        from smartcash.ui.core.errors.handlers import get_error_handler as _get_error_handler
        return _get_error_handler()
    except ImportError:
        return None

# Public exports - only include what's safely importable
__all__ = [
    'ErrorLevel',
    'ErrorContext', 
    'get_error_handler',
]
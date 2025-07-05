"""
File: smartcash/ui/core/errors/exceptions.py
Deskripsi: Definisi hierarki exception untuk komponen UI SmartCash
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type


@dataclass
class ErrorContext:
    """Context information for better error handling and reporting in the UI layer.
    
    This is a UI-specific version of the common ErrorContext, defined here
    to avoid circular imports with the common package.
    """
    component: str = ""
    operation: str = ""
    details: Optional[Dict[str, Any]] = None
    ui_components: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            'component': self.component,
            'operation': self.operation,
            'details': self.details or {}
        }
    
    def add_ui_component(self, name: str, component: Any) -> 'ErrorContext':
        """Add a UI component to the context.
        
        Args:
            name: Name to identify the component
            component: The UI component to add
            
        Returns:
            Self for method chaining
        """
        if self.ui_components is None:
            self.ui_components = {}
        self.ui_components[name] = component
        return self


class SmartCashUIError(Exception):
    """Base exception for all SmartCash UI errors."""
    def __init__(
        self, 
        message: str = "Terjadi error pada UI SmartCash",
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext()
        super().__init__(self.message)
        
    def with_context(self, **kwargs) -> 'SmartCashUIError':
        """Add context to the error and return self for chaining."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        return self


class UIError(SmartCashUIError):
    """Base exception for UI-related errors."""
    def __init__(
        self,
        message: str = "Terjadi kesalahan pada antarmuka pengguna",
        component: str = "unknown",
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        context.component = component or context.component or "unknown"
        super().__init__(message, **kwargs, context=context)


class UIComponentError(UIError):
    """Error related to UI component initialization or operation."""
    def __init__(
        self,
        message: str = "Kesalahan komponen antarmuka",
        component_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if component_type:
            context.details = {**(context.details or {}), 'component_type': component_type}
        super().__init__(message, **kwargs, context=context)


class UIActionError(UIError):
    """Error during UI action execution."""
    def __init__(
        self,
        message: str = "Gagal menjalankan aksi",
        action: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if action:
            context.operation = action
        super().__init__(message, **kwargs, context=context)


# Common exceptions that can be used in the UI layer
class ConfigError(SmartCashUIError):
    """Error in configuration."""
    def __init__(self, message="Error pada konfigurasi UI SmartCash"):
        super().__init__(message)


class ValidationError(SmartCashUIError):
    """Error in input validation."""
    def __init__(self, message="Error validasi input"):
        super().__init__(message)


class NotSupportedError(SmartCashUIError):
    """Feature not supported."""
    def __init__(self, message="Fitur ini tidak didukung"):
        super().__init__(message)

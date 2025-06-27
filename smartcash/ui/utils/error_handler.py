"""
File: smartcash/ui/utils/error_handler.py
Deskripsi: Centralized error handling for UI components with SmartCash integration
"""
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from functools import wraps
import traceback
from dataclasses import asdict

from smartcash.common.exceptions import (
    SmartCashError,
    UIError,
    ErrorContext
)

T = TypeVar('T')

class ErrorHandler:
    """Centralized error handling for UI components."""
    
    def __init__(self, logger_bridge=None, default_component: str = "ui"):
        """Initialize the error handler.
        
        Args:
            logger_bridge: Logger bridge instance for logging errors
            default_component: Default component name for error context
        """
        self.logger_bridge = logger_bridge
        self.default_component = default_component
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        ui_components: Optional[Dict[str, Any]] = None,
        show_ui: bool = True,
        log_level: str = "error"
    ) -> None:
        """Handle an error with proper logging and UI feedback.
        
        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
            ui_components: Dictionary of UI components for showing errors
            show_ui: Whether to show the error in the UI
            log_level: Log level ('error', 'warning', 'info', 'debug')
        """
        # Ensure we have a context
        if context is None:
            context = ErrorContext(component=self.default_component)
        
        # Log the error
        self._log_error(error, context, log_level)
        
        # Show error in UI if requested and components are available
        if show_ui and ui_components:
            self._show_ui_error(error, context, ui_components)
    
    def _log_error(
        self,
        error: Exception,
        context: ErrorContext,
        log_level: str
    ) -> None:
        """Log the error with context information."""
        if not self.logger_bridge:
            return
            
        # Prepare log message
        log_method = getattr(self.logger_bridge, log_level, self.logger_bridge.error)
        
        # Format error details
        error_details = {
            'error_type': error.__class__.__name__,
            'message': str(error),
            'context': asdict(context)
        }
        
        # Add stack trace for non-SmartCash errors
        if not isinstance(error, SmartCashError):
            error_details['stack_trace'] = traceback.format_exc()
        
        # Log the error
        log_method("Error occurred", extra=error_details)
    
    def _show_ui_error(
        self,
        error: Exception,
        context: ErrorContext,
        ui_components: Dict[str, Any]
    ) -> None:
        """Display error in the UI."""
        try:
            # Get UI components with fallbacks
            status_panel = ui_components.get('status_panel')
            log_output = ui_components.get('log_output')
            
            # Format error message
            error_message = f"❌ {str(error)}"
            if context.component:
                error_message = f"[{context.component.upper()}] {error_message}"
            
            # Update status panel if available
            if hasattr(status_panel, 'update_status'):
                status_panel.update_status(error_message, 'error')
            
            # Log to UI console if available
            if hasattr(log_output, 'append_stdout'):
                log_output.append_stdout(f"{error_message}\n")
                
                # Add context details if available
                if context.details:
                    details = ", ".join(f"{k}: {v}" for k, v in context.details.items())
                    log_output.append_stdout(f"  Details: {details}\n")
                
                # Add stack trace for non-SmartCash errors
                if not isinstance(error, SmartCashError):
                    log_output.append_stderr(f"\nStack trace:\n{traceback.format_exc()}")
                    
        except Exception as e:
            # If we can't show the error in the UI, at least log it
            if self.logger_bridge:
                self.logger_bridge.error(
                    "Failed to display error in UI",
                    extra={'error': str(e), 'original_error': str(error)}
                )
    
    def wrap_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for async functions to handle errors consistently."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.handle_error(e)
                raise
        return wrapper
    
    def wrap_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for sync functions to handle errors consistently."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle_error(e)
                raise
        return wrapper

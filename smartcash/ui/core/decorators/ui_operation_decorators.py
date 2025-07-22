"""
Centralized UI Operation Decorators

This module consolidates UI-specific operation decorators used across SmartCash,
providing consistent patterns for widget operations, progress tracking, and
component access.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
from functools import wraps

from smartcash.ui.core.error_utils import with_error_handling as core_with_error_handling
from smartcash.ui.core.errors.enums import ErrorLevel
from smartcash.ui.core.errors.validators import safe_component_operation as core_safe_component_operation

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def safe_widget_operation(
    widget_key_arg: int = 0, 
    widget_key_kwarg: str = "ui_components"
) -> Callable[[F], F]:
    """Decorator for safely performing operations on UI widgets.
    
    This decorator handles operations on UI widgets safely using the centralized
    error handling system, with proper error context for widget operations.
    
    Args:
        widget_key_arg: Position of the UI components dictionary in args
        widget_key_kwarg: Name of the UI components keyword argument
        
    Returns:
        Decorated function that safely handles widget operations
    """
    def decorator(func: F) -> F:
        @wraps(func)
        @core_with_error_handling(
            error_message=lambda e, *a, **kw: (
                f"Error in widget operation {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            reraise=False,
            default=None
        )
        def wrapper(self, *args, **kwargs):
            # Get UI components from args or kwargs
            ui_components = None
            if len(args) > widget_key_arg:
                ui_components = args[widget_key_arg]
            elif widget_key_kwarg in kwargs:
                ui_components = kwargs[widget_key_kwarg]
            
            if not ui_components or not isinstance(ui_components, dict):
                get_error_handler().handle_warning(
                    f"No valid UI components for {func.__name__}",
                    context={
                        'function': func.__name__,
                        'widget_key_arg': widget_key_arg,
                        'widget_key_kwarg': widget_key_kwarg,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                return None
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def safe_progress_operation(
    progress_key: str = "progress_tracker", 
    fallback_keys: List[str] = None
) -> Callable[[F], F]:
    """Decorator for safely performing operations on progress trackers.
    
    This decorator handles progress tracker operations safely using the
    centralized error handling system, with fallbacks to standard progress bars
    if the tracker is not available.
    
    Args:
        progress_key: Key for the progress tracker in UI components
        fallback_keys: List of fallback keys to try if progress_key is not available
        
    Returns:
        Decorated function that safely handles progress operations
    """
    if fallback_keys is None:
        fallback_keys = ["progress_bar", "progress_container"]
    
    def decorator(func: F) -> F:
        @wraps(func)
        @core_safe_component_operation(
            error_message=lambda e, *a, **kw: (
                f"Error in progress operation {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            default=None
        )
        def wrapper(self, *args, **kwargs):
            # Add progress context for better error reporting
            context = {
                'progress_key': progress_key,
                'fallback_keys': fallback_keys,
                'function': func.__name__
            }
            
            # Set context for error reporting
            error_handler = get_error_handler()
            with error_handler.context(**context):
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator

def safe_component_access(
    component_type: str = None, 
    default_value: Any = None
) -> Callable[[F], F]:
    """Decorator for safely accessing and using UI components.
    
    This decorator provides safe access to UI components using the centralized
    error handling system, with proper context about the component being accessed.
    
    Args:
        component_type: Type of component being accessed (for error context)
        default_value: Default value to return if component access fails
        
    Returns:
        Decorated function that safely handles component access
    """
    def decorator(func: F) -> F:
        @wraps(func)
        @core_safe_component_operation(
            error_message=lambda e, *a, **kw: (
                f"Error accessing component {component_type or 'unknown'} in {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            default=default_value
        )
        def wrapper(self, *args, **kwargs):
            # Add component context for better error reporting
            context = {
                'component_type': component_type,
                'function': func.__name__,
                'default_value': default_value
            }
            
            # Set context for error reporting
            error_handler = get_error_handler()
            with error_handler.context(**context):
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator

def safe_button_operation(
    button_name: str = None,
    disable_during_operation: bool = True,
    status_update_key: str = "header_container"
) -> Callable[[F], F]:
    """Decorator for safely handling button click operations.
    
    This decorator provides safe button handling with automatic disable/enable
    during operations and status updates.
    
    Args:
        button_name: Name of the button for context
        disable_during_operation: Whether to disable button during operation
        status_update_key: Key for status update component
        
    Returns:
        Decorated function that safely handles button operations
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            button_ref = None
            status_component = None
            
            try:
                # Get button reference if available
                if hasattr(self, 'get_component'):
                    button_ref = self.get_component(button_name) if button_name else None
                    status_component = self.get_component(status_update_key)
                
                # Disable button during operation
                if disable_during_operation and button_ref and hasattr(button_ref, 'disabled'):
                    button_ref.disabled = True
                
                # Update status if available
                if status_component and hasattr(status_component, 'update_status'):
                    operation_name = button_name or func.__name__
                    status_component.update_status(f"Processing {operation_name}...", "info")
                
                # Execute the operation
                result = func(self, *args, **kwargs)
                
                # Update success status
                if status_component and hasattr(status_component, 'update_status'):
                    status_component.update_status(f"{operation_name} completed", "success")
                
                return result
                
            except Exception as e:
                # Update error status
                if status_component and hasattr(status_component, 'update_status'):
                    operation_name = button_name or func.__name__
                    status_component.update_status(f"{operation_name} failed: {e}", "error")
                
                # Handle the error through centralized system
                get_error_handler().handle_error(
                    error_msg=f"Button operation {button_name or func.__name__} failed: {e}",
                    level=ErrorLevel.ERROR,
                    exc_info=True,
                    create_ui_error=True
                )
                
                raise
                
            finally:
                # Re-enable button
                if disable_during_operation and button_ref and hasattr(button_ref, 'disabled'):
                    button_ref.disabled = False
                
        return wrapper
    return decorator

def safe_form_operation(
    form_key: str = "form_container",
    validation_required: bool = True
) -> Callable[[F], F]:
    """Decorator for safely handling form operations.
    
    This decorator provides safe form handling with optional validation
    and error reporting through the centralized system.
    
    Args:
        form_key: Key for the form container component
        validation_required: Whether to perform validation before operation
        
    Returns:
        Decorated function that safely handles form operations
    """
    def decorator(func: F) -> F:
        @wraps(func)
        @core_safe_component_operation(
            error_message=lambda e, *a, **kw: (
                f"Error in form operation {func.__name__}: {str(e)}"
            ),
            level=ErrorLevel.WARNING,
            default=None
        )
        def wrapper(self, *args, **kwargs):
            # Add form context for better error reporting
            context = {
                'form_key': form_key,
                'validation_required': validation_required,
                'function': func.__name__
            }
            
            # Get form component if available
            form_component = None
            if hasattr(self, 'get_component'):
                form_component = self.get_component(form_key)
            
            # Perform validation if required
            if validation_required and form_component:
                if hasattr(form_component, 'validate'):
                    validation_result = form_component.validate()
                    if not validation_result.get('valid', True):
                        error_msg = validation_result.get('error', 'Form validation failed')
                        get_error_handler().handle_error(
                            error_msg=f"Form validation failed: {error_msg}",
                            level=ErrorLevel.WARNING,
                            create_ui_error=True
                        )
                        return {'success': False, 'error': error_msg}
            
            # Set context for error reporting
            error_handler = get_error_handler()
            with error_handler.context(**context):
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator
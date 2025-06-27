"""
File: smartcash/ui/utils/error_utils.py
Deskripsi: Fixed error utilities dengan parameter conflict resolution dan improved decorator
"""

from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
import traceback
import logging

from smartcash.common.exceptions import ErrorContext, SmartCashError

T = TypeVar('T')

def create_error_context(
    component: str = "",
    operation: str = "",
    details: Optional[Dict[str, Any]] = None,
    ui_components: Optional[Dict[str, Any]] = None
) -> ErrorContext:
    """
    Create error context dengan parameter yang jelas dan tidak konfliks
    
    Args:
        component: Nama komponen yang mengalami error
        operation: Operasi yang sedang berjalan
        details: Detail tambahan error
        ui_components: UI components untuk error display
        
    Returns:
        ErrorContext instance
    """
    return ErrorContext(
        component=component,
        operation=operation,
        details=details or {},
        ui_components=ui_components
    )

def error_handler_scope(
    component: str = "ui",
    operation: str = "unknown",
    logger: Optional[logging.Logger] = None,
    ui_components: Optional[Dict[str, Any]] = None,
    show_ui_error: bool = True
):
    """
    FIXED: Context manager untuk error handling scope dengan proper parameter handling
    
    Args:
        component: Nama komponen
        operation: Nama operasi
        logger: Logger instance (opsional)
        ui_components: UI components untuk error display
        show_ui_error: Tampilkan error di UI
    """
    class ErrorScope:
        def __init__(self):
            self.context = create_error_context(
                component=component,
                operation=operation,
                ui_components=ui_components
            )
            self.logger = logger or logging.getLogger(__name__)
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_value, exc_traceback):
            if exc_type is not None:
                # Log error dengan context
                error_msg = f"[{component}:{operation}] {str(exc_value)}"
                self.logger.error(error_msg, exc_info=(exc_type, exc_value, exc_traceback))
                
                # Show UI error jika diminta
                if show_ui_error and ui_components:
                    self._show_ui_error(str(exc_value))
                    
                # Suppress exception untuk continue execution
                return True
            return False
            
        def _show_ui_error(self, error_message: str):
            """Show error di UI dengan safe fallback"""
            try:
                from smartcash.ui.utils.fallback_utils import show_error_ui
                show_error_ui(ui_components, error_message)
            except Exception:
                # Fallback ke console jika UI error
                print(f"🚨 [UI Error] {error_message}")
    
    return ErrorScope()

def with_error_handling(
    error_handler: Optional[Any] = None,
    component: str = "ui",
    operation: str = "unknown",
    show_traceback: bool = False,
    ui_components: Optional[Dict[str, Any]] = None,
    fallback_value: Any = None
) -> Callable:
    """
    FIXED: Decorator dengan parameter conflict resolution dan improved error handling
    
    Args:
        error_handler: Error handler instance (opsional)
        component: Nama komponen yang menggunakan decorator
        operation: Nama operasi yang di-wrap
        show_traceback: Tampilkan traceback dalam error
        ui_components: UI components untuk error display
        fallback_value: Nilai fallback jika terjadi error
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # FIXED: Create context tanpa parameter conflict
                error_context = create_error_context(
                    component=component,
                    operation=operation,
                    details={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    },
                    ui_components=ui_components
                )
                
                # Use error handler jika tersedia
                if error_handler and hasattr(error_handler, 'handle_error'):
                    error_handler.handle_error(
                        error=e,
                        context=error_context,
                        ui_components=ui_components,
                        show_ui=True
                    )
                else:
                    # Fallback error handling
                    _fallback_error_handling(e, error_context, show_traceback)
                
                # Return fallback value atau re-raise
                if fallback_value is not None:
                    return fallback_value
                    
                # Re-raise untuk critical errors
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                    
                # Return None untuk non-critical errors
                return None
                
        return wrapper
    return decorator

def _fallback_error_handling(
    error: Exception, 
    context: ErrorContext, 
    show_traceback: bool = False
) -> None:
    """
    Fallback error handling ketika error handler tidak tersedia
    
    Args:
        error: Exception yang terjadi
        context: Error context
        show_traceback: Tampilkan traceback
    """
    # Format error message dengan context
    error_msg = f"🚨 [{context.component}:{context.operation}] {str(error)}"
    
    # Print error ke console
    print(error_msg)
    
    # Print traceback jika diminta
    if show_traceback:
        print("📋 Traceback:")
        traceback.print_exc()
    
    # Show UI error jika UI components tersedia
    if context.ui_components:
        try:
            from smartcash.ui.utils.fallback_utils import show_error_ui
            show_error_ui(context.ui_components, str(error))
        except Exception:
            pass  # Silent fail untuk UI error

def log_errors(
    logger: Optional[logging.Logger] = None,
    level: str = "error",
    component: str = "ui",
    operation: str = "unknown"
) -> Callable:
    """
    Simple error logging decorator tanpa UI integration
    
    Args:
        logger: Logger instance
        level: Log level (error, warning, info)
        component: Nama komponen
        operation: Nama operasi
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(__name__)
                log_method = getattr(log, level.lower(), log.error)
                
                error_msg = f"[{component}:{operation}] {func.__name__}: {str(e)}"
                log_method(error_msg, exc_info=True)
                
                raise  # Re-raise untuk handle di level atas
                
        return wrapper
    return decorator

# FIXED: Safe error context creation tanpa parameter conflicts
def safe_create_context(comp: str, op: str, **kwargs) -> ErrorContext:
    """Safe error context creation dengan parameter aliases"""
    return create_error_context(
        component=comp,
        operation=op,
        details=kwargs.get('details'),
        ui_components=kwargs.get('ui_components')
    )

# One-liner utilities untuk common error patterns
handle_ui_error = lambda ui_components, error_msg: show_error_ui(ui_components, error_msg) if ui_components else print(f"🚨 {error_msg}")
log_and_ignore = lambda logger, error, msg="": logger.error(f"{msg}: {str(error)}") if logger else print(f"🚨 {msg}: {str(error)}")
safe_execute = lambda func, fallback=None: fallback if _safe_call(func, fallback) is None else _safe_call(func, fallback)

def _safe_call(func: Callable, fallback: Any = None) -> Any:
    """Safe function call dengan fallback"""
    try:
        return func()
    except Exception:
        return fallback

# Export untuk backward compatibility
__all__ = [
    'create_error_context',
    'error_handler_scope', 
    'with_error_handling',
    'log_errors',
    'safe_create_context',
    'handle_ui_error',
    'log_and_ignore',
    'safe_execute'
]
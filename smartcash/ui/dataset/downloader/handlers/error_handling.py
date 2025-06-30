"""
File: smartcash/ui/dataset/downloader/error_handling.py
Deskripsi: Centralized error handling for the downloader module
"""

from typing import Any, Dict, Optional, Callable, TypeVar, Type
from functools import wraps

from smartcash.common.exceptions import SmartCashError
from smartcash.ui.pretrained.utils import (
    with_error_handling,
    log_errors,
    get_logger,
    create_error_context,
    ErrorContext,
    safe_ui_operation
)

# Type variable for generic function return type
T = TypeVar('T')

def create_downloader_error_context(
    operation: str = "",
    details: Optional[Dict[str, Any]] = None,
    ui_components: Optional[Dict[str, Any]] = None
) -> ErrorContext:
    """
    Create error context for downloader operations
    
    Args:
        operation: The operation being performed
        details: Additional error details
        ui_components: UI components for error display
        
    Returns:
        ErrorContext instance
    """
    return create_error_context(
        component="downloader",
        operation=operation,
        details=details or {},
        ui_components=ui_components
    )

def with_downloader_error_handling(
    operation: str = "unknown",
    show_traceback: bool = False,
    fallback_value: Any = None
):
    """
    Decorator for downloader operations with error handling
    
    Args:
        operation: Name of the operation for logging
        show_traceback: Whether to include traceback in error messages
        fallback_value: Value to return on error if not None
    """
    return with_error_handling(
        component="downloader",
        operation=operation,
        show_traceback=show_traceback,
        fallback_value=fallback_value
    )

def handle_downloader_error(
    error: Exception,
    context: Optional[ErrorContext] = None,
    logger=None,
    ui_components: Optional[Dict[str, Any]] = None
) -> None:
    """
    Centralized error handler for downloader operations
    
    Args:
        error: The exception that was raised
        context: Error context (will be created if None)
        logger: Logger instance for error logging
        ui_components: UI components for error display
    """
    if context is None:
        context = create_downloader_error_context(
            operation="unknown",
            ui_components=ui_components
        )
    
    # Log the error
    error_msg = f"Downloader error in {context.operation}: {str(error)}"
    if logger:
        logger.error(error_msg, exc_info=True)
    
    # Update UI if components are available
    if ui_components and 'error_output' in ui_components:
        from smartcash.ui.dataset.downloader.utils.ui_utils import log_to_accordion
        log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
    
    # Re-raise as SmartCashError if it isn't already
    if not isinstance(error, SmartCashError):
        raise SmartCashError(error_msg) from error

# Common error handling decorators
downloader_operation = with_downloader_error_handling
downloader_ui_operation = safe_ui_operation(component="downloader")
downloader_log_errors = log_errors(component="downloader")

# Common error contexts
DOWNLOAD_CONTEXT = create_downloader_error_context(operation="download_dataset")
CLEANUP_CONTEXT = create_downloader_error_context(operation="cleanup_dataset")
VALIDATE_CONTEXT = create_downloader_error_context(operation="validate_config")

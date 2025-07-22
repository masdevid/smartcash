"""
SmartCash UI Core Decorators Module

This module provides centralized decorators for consistent error handling,
logging, and UI operations across all SmartCash modules.

All decorators are consolidated here to eliminate duplication and provide
a single import point for all UI decorator functionality.
"""

# Error handling decorators
from .error_decorators import (
    handle_errors,
    handle_ui_errors,
    safe_ui_operation,
    log_errors,
    suppress_errors,
    retry_on_failure,
    safe_component_operation
)

# Log suppression decorators  
from .log_decorators import (
    suppress_ui_init_logs,
    suppress_all_init_logs,
    suppress_initial_logs,
    SuppressInitialLogs,
    activate_log_suppression,
    deactivate_log_suppression,
    is_suppression_active,
    extend_suppression
)

# UI operation decorators
from .ui_operation_decorators import (
    safe_widget_operation,
    safe_progress_operation,
    safe_component_access,
    safe_button_operation,
    safe_form_operation
)

# Legacy decorators removed - all now centralized in this module

__all__ = [
    # Error handling
    'handle_errors',
    'handle_ui_errors', 
    'safe_ui_operation',
    'log_errors',
    'suppress_errors',
    'retry_on_failure',
    'safe_component_operation',
    
    # Log suppression
    'suppress_ui_init_logs',
    'suppress_all_init_logs',
    'suppress_initial_logs',
    'SuppressInitialLogs',
    'activate_log_suppression',
    'deactivate_log_suppression',
    'is_suppression_active',
    'extend_suppression',
    
    # UI operations
    'safe_widget_operation',
    'safe_progress_operation',
    'safe_component_access',
    'safe_button_operation',
    'safe_form_operation',
    
    # All decorators are now centralized here
]

"""
File: smartcash/ui/setup/dependency/utils/ui/state.py

UI state management utilities.

This module provides utilities for managing UI state, progress tracking, and status updates.
"""

# Standard library imports
import functools
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    'ProgressSteps',
    'create_operation_context',
    'update_status_panel',
    'log_to_ui_safe',
    'update_progress_step',
    'show_progress_tracker_safe',
    'reset_progress_tracker_safe',
    'complete_operation_with_message',
    'error_operation_with_message',
    'update_package_status_by_name',
    'batch_update_package_status',
    'with_button_context'
]

class ProgressSteps(Enum):
    """Progress steps enumeration"""
    INIT = "init"
    ANALYSIS = "analysis" 
    INSTALLATION = "installation"
    COMPLETE = "complete"

def create_operation_context(ui_components: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
    """Create operation context untuk tracking"""
    return {
        'operation_name': operation_name,
        'ui_components': ui_components,
        'start_time': __import__('time').time()
    }

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan message"""
    try:
        from smartcash.ui.components import update_status_panel as ui_update_status
        status_panel = ui_components.get('status_panel')
        if status_panel:
            ui_update_status(status_panel, message, status_type)
    except:
        pass

def log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Safe logging ke UI output"""
    try:
        logger = ui_components.get('logger')
        if logger:
            getattr(logger, level, logger.info)(message)
    except:
        pass

def update_progress_step(ui_components: Dict[str, Any], progress_type: str, value: int, message: str = "", color: str = None):
    """Update progress step dengan safe handling"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update'):
            progress_tracker.update(progress_type, value, message)
    except:
        pass

def show_progress_tracker_safe(ui_components: Dict[str, Any], operation_name: str):
    """Safely show progress tracker"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show(operation=operation_name)
    except:
        pass

def reset_progress_tracker_safe(ui_components: Dict[str, Any]):
    """Safely reset progress tracker"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    except:
        pass

def complete_operation_with_message(ui_components: Dict[str, Any], message: str):
    """Complete operation dengan success message"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        update_status_panel(ui_components, message, "success")
    except:
        pass

def error_operation_with_message(ui_components: Dict[str, Any], message: str):
    """Error operation dengan error message"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        update_status_panel(ui_components, message, "error")
    except:
        pass

def update_package_status_by_name(ui_components: Dict[str, Any], package_name: str, status: str):
    """Update package status by name"""
    try:
        # Import locally to avoid circular dependency
        from smartcash.ui.setup.dependency.components.ui_package_selector import update_package_status
        package_selector = ui_components.get('package_selector')
        if package_selector:
            update_package_status(package_selector, package_name, status == 'installed')
    except:
        pass

def batch_update_package_status(ui_components: Dict[str, Any], status_mapping: Dict[str, str]):
    """Batch update package status"""
    for package_name, status in status_mapping.items():
        update_package_status_by_name(ui_components, package_name, status)

def with_button_context(ui_components: Dict[str, Any], button_key: str):
    """Decorator untuk disable/enable button during operation"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            button = ui_components.get(button_key)
            original_disabled = getattr(button, 'disabled', False) if button else False
            
            # Disable button
            if button and hasattr(button, 'disabled'):
                button.disabled = True
            
            try:
                return func(*args, **kwargs)
            finally:
                # Re-enable button
                if button and hasattr(button, 'disabled'):
                    button.disabled = original_disabled
        
        return wrapper
    return decorator
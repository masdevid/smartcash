"""
File: smartcash/ui/dataset/augmentation/utils/ui_logger_utils.py
Deskripsi: Unified logging utility untuk semua augmentation handlers
"""

from typing import Dict, Any

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info', prefix: str = ""):
    """Unified logging ke UI dengan fallback chain"""
    formatted_message = f"{prefix}{message}" if prefix else message
    
    # Priority 1: UI Logger
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(formatted_message)
        return
    
    # Priority 2: Widget Display
    try:
        from IPython.display import display, HTML
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = color_map.get(level, '#007bff')
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{formatted_message}</div>'
            with widget:
                display(HTML(html))
            return
    except Exception:
        pass
    
    # Priority 3: Console fallback
    print(formatted_message)

def show_progress_safe(ui_components: Dict[str, Any], operation: str):
    """Safe progress show dengan fallback"""
    try:
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'show'):
            tracker.show(operation)
    except Exception:
        pass

def complete_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress complete dengan fallback"""
    try:
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'complete'):
            tracker.complete(message)
    except Exception:
        pass

def error_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress error dengan fallback"""
    try:
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'error'):
            tracker.error(message)
    except Exception:
        pass

# One-liner utilities dengan unified logging
log_info = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'info', prefix)
log_success = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'success', prefix)
log_warning = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'warning', prefix)
log_error = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'error', prefix)
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

def show_progress_safe(ui_components: Dict[str, Any], operation: str, steps=None, step_weights=None):
    """Safe progress show dengan fallback dan API yang benar"""
    try:
        # Gunakan progress_tracker dengan API yang benar
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            # Gunakan metode show dengan parameter yang benar
            if steps and step_weights:
                progress_tracker.show(operation, steps, step_weights)
            else:
                # Fallback dengan default steps jika tidak disediakan
                default_steps = ["prepare", "process", "verify"]
                default_weights = {"prepare": 20, "process": 60, "verify": 20}
                progress_tracker.show(operation, default_steps, default_weights)
        else:
            # Fallback ke metode lama atau tracker lama
            tracker = ui_components.get('tracker')
            if tracker and hasattr(tracker, 'show'):
                tracker.show(operation)
            else:
                show_fn = ui_components.get('show_for_operation')
                if show_fn:
                    show_fn(operation)
    except Exception:
        pass

def complete_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress complete dengan fallback dan API yang benar"""
    try:
        # Gunakan progress_tracker dengan API yang benar
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        else:
            # Fallback ke metode lama atau tracker lama
            tracker = ui_components.get('tracker')
            if tracker and hasattr(tracker, 'complete'):
                tracker.complete(message)
            else:
                complete_fn = ui_components.get('complete_operation')
                if complete_fn:
                    complete_fn(message)
    except Exception:
        pass

def error_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress error dengan fallback dan API yang benar"""
    try:
        # Gunakan progress_tracker dengan API yang benar
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        else:
            # Fallback ke metode lama atau tracker lama
            tracker = ui_components.get('tracker')
            if tracker and hasattr(tracker, 'error'):
                tracker.error(message)
            else:
                error_fn = ui_components.get('error_operation')
                if error_fn:
                    error_fn(message)
    except Exception:
        pass

# One-liner utilities dengan unified logging
log_info = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'info', prefix)
log_success = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'success', prefix)
log_warning = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'warning', prefix)
log_error = lambda ui_components, msg, prefix="": log_to_ui(ui_components, msg, 'error', prefix)
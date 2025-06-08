"""
File: smartcash/ui/dataset/augmentation/utils/ui_utils.py
Deskripsi: UI utilities dengan unified logging dan progress management
"""

from typing import Dict, Any

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Unified logging ke UI dengan fallback chain"""
    try:
        # Priority 1: UI Logger
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Priority 2: Log widget
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            from IPython.display import display, HTML
            color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = color_map.get(level, '#007bff')
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
            
            with widget:
                display(HTML(html))
            return
            
    except Exception:
        pass
    
    # Fallback
    print(message)

def show_progress_safe(ui_components: Dict[str, Any], operation: str):
    """Safe progress show untuk operation"""
    try:
        config = ui_components.get('config', {})
        progress_config = config.get('progress', {}).get('operations', {})
        
        op_config = progress_config.get(operation, {
            'steps': ["prepare", "process", "complete"],
            'weights': {"prepare": 20, "process": 60, "complete": 20}
        })
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show(
                operation=operation.replace('_', ' ').title(),
                steps=op_config.get('steps'),
                step_weights=op_config.get('weights')
            )
        elif 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
    except Exception:
        pass

def complete_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress complete"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        elif 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
    except Exception:
        pass

def error_progress_safe(ui_components: Dict[str, Any], message: str):
    """Safe progress error"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        elif 'error_operation' in ui_components:
            ui_components['error_operation'](message)
    except Exception:
        pass

def clear_outputs(ui_components: Dict[str, Any]):
    """Clear output areas"""
    # Reset progress
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'reset'):
        progress_tracker.reset()
    elif 'reset_all' in ui_components:
        ui_components['reset_all']()
    
    # Clear confirmation area
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output(wait=True)

def get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Get widget value dengan type safety"""
    widget = ui_components.get(key)
    if widget and hasattr(widget, 'value'):
        try:
            value = widget.value
            if isinstance(default, int) and isinstance(value, (int, float)):
                return int(value)
            elif isinstance(default, float) and isinstance(value, (int, float)):
                return float(value)
            return value
        except Exception:
            pass
    return default

def extract_augmentation_types(ui_components: Dict[str, Any]) -> list:
    """Extract augmentation types dari UI"""
    for widget_key in ['augmentation_types', 'aug_options', 'types_widget']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value') and widget.value:
            return list(widget.value)
    return ['combined']

# One-liner utilities
log_info = lambda ui_components, msg: log_to_ui(ui_components, msg, 'info')
log_success = lambda ui_components, msg: log_to_ui(ui_components, msg, 'success')
log_warning = lambda ui_components, msg: log_to_ui(ui_components, msg, 'warning')
log_error = lambda ui_components, msg: log_to_ui(ui_components, msg, 'error')
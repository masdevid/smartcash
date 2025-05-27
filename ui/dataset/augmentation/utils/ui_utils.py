"""
File: smartcash/ui/dataset/augmentation/utils/ui_utils.py
Deskripsi: Shared utilities untuk augmentation UI handlers
"""

from typing import Dict, Any, List, Optional

def extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Extract augmentation types dari berbagai widget structures"""
    for widget_key in ['augmentation_types', 'aug_options', 'types_widget']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value') and widget.value:
            return list(widget.value)
        
        # Try container with children
        if hasattr(widget, 'children') and widget.children:
            for child in widget.children:
                if hasattr(child, 'value') and child.value:
                    return list(child.value)
    
    return ['combined']  # Default fallback

def get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Safely get widget value dengan fallback"""
    widget = ui_components.get(key)
    return getattr(widget, 'value', default) if widget else default

def log_to_ui_only(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log message hanya ke UI log_output, tidak ke console"""
    logger = ui_components.get('logger')
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)

def reset_ui_state(ui_components: Dict[str, Any]):
    """Reset progress tracker dan log output"""
    # Reset progress tracker
    tracker = ui_components.get('tracker')
    if tracker and hasattr(tracker, 'reset'):
        tracker.reset()
    
    # Clear log output
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)

def create_service_safely(ui_components: Dict[str, Any]):
    """Create augmentor service dengan error handling"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        return create_service_from_ui(ui_components)
    except ImportError as e:
        log_to_ui_only(ui_components, f"❌ Service import error: {str(e)}", 'error')
        raise

def enforce_train_split(ui_components: Dict[str, Any]):
    """Force train split di target_split widget"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        target_split_widget.value = 'train'

def create_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk service integration"""
    def callback(step: str, current: int, total: int, message: str):
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            percentage = min(100, int((current / max(1, total)) * 100))
            tracker.update(step, percentage, message)
    return callback

# One-liner utilities
show_progress = lambda ui_components, operation: getattr(ui_components.get('tracker'), 'show', lambda x: None)(operation)
complete_progress = lambda ui_components, message: getattr(ui_components.get('tracker'), 'complete', lambda m: None)(message)
error_progress = lambda ui_components, message: getattr(ui_components.get('tracker'), 'error', lambda m: None)(message)
update_progress = lambda ui_components, step, pct, msg: getattr(ui_components.get('tracker'), 'update', lambda s, p, m: None)(step, pct, msg)
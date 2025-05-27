"""
File: smartcash/ui/dataset/augmentation/utils/ui_utils.py
Deskripsi: UI utilities dengan unified logging dan service integration
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui

def extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Extract augmentation types dengan service support"""
    for widget_key in ['augmentation_types', 'aug_options', 'types_widget']:
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value') and widget.value:
            return list(widget.value)
        
        if hasattr(widget, 'children') and widget.children:
            for child in widget.children:
                if hasattr(child, 'value') and child.value:
                    return list(child.value)
    
    return ['combined']

def get_widget_value_safe(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Get widget value dengan service type safety"""
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

def reset_ui_state(ui_components: Dict[str, Any]):
    """Reset UI state untuk service operations"""
    tracker = ui_components.get('tracker')
    if tracker and hasattr(tracker, 'reset'):
        tracker.reset()
    
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)
    
    ui_components.pop('service_result', None)

def create_service_safely(ui_components: Dict[str, Any]):
    """Create service dengan comprehensive error handling"""
    try:
        from smartcash.dataset.augmentor.service import create_service_from_ui
        
        service = create_service_from_ui(ui_components)
        ui_components['service'] = service
        return service
        
    except ImportError as e:
        log_to_ui(ui_components, f"Service import error: {str(e)}", 'error', "❌ ")
        raise
    except Exception as e:
        log_to_ui(ui_components, f"Service creation error: {str(e)}", 'error', "❌ ")
        raise

def enforce_train_split(ui_components: Dict[str, Any]):
    """Force train split untuk service compatibility"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        target_split_widget.value = 'train'

def create_service_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk service integration"""
    def callback(step: str, current: int, total: int, message: str):
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            percentage = min(100, int((current / max(1, total)) * 100))
            tracker.update(step, percentage, f"🎯 {message}")
    return callback

def get_service_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get service status dengan fallback"""
    service = ui_components.get('service')
    if service and hasattr(service, 'get_augmentation_status'):
        try:
            return service.get_augmentation_status()
        except Exception as e:
            log_to_ui(ui_components, f"Error getting service status: {str(e)}", 'error', "❌ ")
    
    return {'status': 'not_available', 'service_ready': False}

def validate_service_config(ui_components: Dict[str, Any]) -> bool:
    """Validate UI config untuk service compatibility"""
    try:
        from smartcash.dataset.augmentor.config import validate_ui_parameters
        return validate_ui_parameters(ui_components)
    except ImportError:
        return all([
            get_widget_value_safe(ui_components, 'num_variations', 0) > 0,
            get_widget_value_safe(ui_components, 'target_count', 0) > 0,
            len(extract_augmentation_types(ui_components)) > 0
        ])

# One-liner utilities dengan unified logging
show_service_progress = lambda ui_components, operation: getattr(ui_components.get('tracker'), 'show', lambda x: None)(f"service_{operation}")
complete_service_progress = lambda ui_components, message: getattr(ui_components.get('tracker'), 'complete', lambda m: None)(f"🎯 {message}")
error_service_progress = lambda ui_components, message: getattr(ui_components.get('tracker'), 'error', lambda m: None)(f"🎯 {message}")
update_service_progress = lambda ui_components, step, pct, msg: getattr(ui_components.get('tracker'), 'update', lambda s, p, m: None)(step, pct, f"🎯 {msg}")

# Service-specific helpers dengan unified logging
get_service_config = lambda ui_components: ui_components.get('service_config', ui_components.get('config', {}))
set_service_result = lambda ui_components, result: ui_components.update({'service_result': result})
get_service_result = lambda ui_components: ui_components.get('service_result', {})
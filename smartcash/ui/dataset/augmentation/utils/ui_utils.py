"""
File: smartcash/ui/dataset/augmentation/utils/ui_utils.py
Deskripsi: UI utilities dengan unified logging dan orchestrator integration
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.dataset.augmentation.utils.ui_logger_utils import log_to_ui

def extract_augmentation_types(ui_components: Dict[str, Any]) -> List[str]:
    """Extract augmentation types dengan orchestrator support"""
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
    """Get widget value dengan orchestrator type safety"""
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
    """Reset UI state untuk orchestrator operations"""
    tracker = ui_components.get('tracker')
    if tracker and hasattr(tracker, 'reset'):
        tracker.reset()
    
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)
    
    ui_components.pop('orchestrator_result', None)

def create_orchestrator_safely(ui_components: Dict[str, Any]):
    """Create orchestrator dengan comprehensive error handling"""
    try:
        from smartcash.dataset.augmentor.services.augmentation_orchestrator import AugmentationOrchestrator
        from smartcash.dataset.augmentor.config import extract_ui_config
        
        config = extract_ui_config(ui_components)
        orchestrator = AugmentationOrchestrator(config, ui_components)
        
        ui_components['orchestrator'] = orchestrator
        return orchestrator
        
    except ImportError as e:
        log_to_ui(ui_components, f"Orchestrator import error: {str(e)}", 'error', "❌ ")
        raise
    except Exception as e:
        log_to_ui(ui_components, f"Orchestrator creation error: {str(e)}", 'error', "❌ ")
        raise

def enforce_train_split(ui_components: Dict[str, Any]):
    """Force train split untuk orchestrator compatibility"""
    target_split_widget = ui_components.get('target_split')
    if target_split_widget and hasattr(target_split_widget, 'value'):
        target_split_widget.value = 'train'

def create_orchestrator_progress_callback(ui_components: Dict[str, Any]):
    """Create progress callback untuk orchestrator integration"""
    def callback(step: str, current: int, total: int, message: str):
        tracker = ui_components.get('tracker')
        if tracker and hasattr(tracker, 'update'):
            percentage = min(100, int((current / max(1, total)) * 100))
            tracker.update(step, percentage, f"🎯 {message}")
    return callback

def get_orchestrator_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get orchestrator status dengan fallback"""
    orchestrator = ui_components.get('orchestrator')
    if orchestrator and hasattr(orchestrator, 'get_augmentation_status'):
        try:
            return orchestrator.get_augmentation_status()
        except Exception as e:
            log_to_ui(ui_components, f"Error getting orchestrator status: {str(e)}", 'error', "❌ ")
    
    return {'status': 'not_available', 'orchestrator_ready': False}

def validate_orchestrator_config(ui_components: Dict[str, Any]) -> bool:
    """Validate UI config untuk orchestrator compatibility"""
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
show_orchestrator_progress = lambda ui_components, operation: getattr(ui_components.get('tracker'), 'show', lambda x: None)(f"orchestrator_{operation}")
complete_orchestrator_progress = lambda ui_components, message: getattr(ui_components.get('tracker'), 'complete', lambda m: None)(f"🎯 {message}")
error_orchestrator_progress = lambda ui_components, message: getattr(ui_components.get('tracker'), 'error', lambda m: None)(f"🎯 {message}")
update_orchestrator_progress = lambda ui_components, step, pct, msg: getattr(ui_components.get('tracker'), 'update', lambda s, p, m: None)(step, pct, f"🎯 {msg}")

# Orchestrator-specific helpers dengan unified logging
get_orchestrator_config = lambda ui_components: ui_components.get('orchestrator_config', ui_components.get('config', {}))
set_orchestrator_result = lambda ui_components, result: ui_components.update({'orchestrator_result': result})
get_orchestrator_result = lambda ui_components: ui_components.get('orchestrator_result', {})
"""
File: smartcash/ui/dataset/augmentation/utils/progress_manager.py
Deskripsi: Manager progress bar untuk augmentasi dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message
from smartcash.ui.dataset.augmentation.utils.notification_manager import notify_progress

def reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """Reset progress bar ke kondisi awal."""
    try:
        if 'progress_bar' in ui_components and ui_components['progress_bar'] is not None:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = "Progress: 0%"
            ui_components['progress_bar'].layout.visibility = 'hidden'
        
        for label_key in ['overall_label', 'step_label']:
            if label_key in ui_components and ui_components[label_key] is not None:
                ui_components[label_key].value = ""
                ui_components[label_key].layout.visibility = 'hidden'
                
        if 'current_progress' in ui_components and ui_components['current_progress'] is not None:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = "Step 0/0"
            ui_components['current_progress'].layout.visibility = 'hidden'
    except Exception as e:
        log_message(ui_components, f"Gagal mereset progress bar: {str(e)}", "warning", "⚠️")

def show_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan progress container dan set progress awal."""
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = 0
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    for label_key in ['progress_message', 'step_label', 'overall_label']:
        if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
            ui_components[label_key].value = message
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'visible'

def update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """Update progress bar dan pesan."""
    value = max(0, min(100, value))
    
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = value
    
    if message:
        for label_key in ['progress_message', 'step_label', 'overall_label']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
                ui_components[label_key].value = message
    
    observer_manager = ui_components.get('observer_manager')
    if observer_manager:
        try:
            observer_id = ui_components.get('observer_group', 'augmentation_progress')
            observer_manager.notify(observer_id, {'progress': value, 'message': message})
        except Exception:
            pass

def setup_multi_progress(ui_components: Dict[str, Any], tracking_keys: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Setup multi-progress tracking."""
    if tracking_keys is None:
        tracking_keys = {
            'overall_progress': 'progress_bar',
            'step_progress': 'progress_bar',
            'overall_label': 'overall_label',
            'step_label': 'step_label'
        }
    
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    if not callable(ui_components.get('update_progress')):
        ui_components['update_progress'] = lambda value, message=None: update_progress(ui_components, value, message)
    
    if not callable(ui_components.get('reset_progress')):
        ui_components['reset_progress'] = lambda: reset_progress_bar(ui_components)
    
    if not callable(ui_components.get('show_progress')):
        ui_components['show_progress'] = lambda message: show_progress(ui_components, message)
    
    return ui_components

def setup_progress_indicator(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress indicator jika progress bar tidak tersedia."""
    if 'progress_bar' not in ui_components:
        from ipywidgets import FloatProgress, Label, VBox, HBox
        
        progress_bar = FloatProgress(value=0, min=0, max=100, description='Loading:')
        progress_message = Label(value='Mempersiapkan...')
        progress_container = VBox([HBox([progress_bar, progress_message])])
        
        ui_components['progress_bar'] = progress_bar
        ui_components['progress_message'] = progress_message
        ui_components['progress_container'] = progress_container
        
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                children = list(ui_components['ui'].children)
                children.append(progress_container)
                ui_components['ui'].children = tuple(children)
            except:
                pass
    
    return ui_components
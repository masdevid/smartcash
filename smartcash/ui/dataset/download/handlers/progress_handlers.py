"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Setup progress tracking dengan komunikasi service-UI yang diperbaiki
"""

from typing import Dict, Any
from smartcash.components.observer import ObserverManager

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking dan observer untuk komunikasi service-UI."""
    
    # Create observer untuk progress updates
    observer_manager = ObserverManager()
    
    # Progress observer untuk overall progress
    def overall_progress_observer(event_type: str, sender: Any, **kwargs):
        progress = kwargs.get('progress', 0)
        message = kwargs.get('message', 'Processing...')
        
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = min(100, max(0, progress))
            ui_components['progress_bar'].description = f"Progress: {progress}%"
            if hasattr(ui_components['progress_bar'], 'layout'):
                ui_components['progress_bar'].layout.visibility = 'visible'
        
        if 'overall_label' in ui_components:
            ui_components['overall_label'].value = message
            if hasattr(ui_components['overall_label'], 'layout'):
                ui_components['overall_label'].layout.visibility = 'visible'
    
    # Step progress observer untuk per-tahap progress  
    def step_progress_observer(event_type: str, sender: Any, **kwargs):
        step_progress = kwargs.get('step_progress', kwargs.get('progress', 0))
        step_message = kwargs.get('step_message', kwargs.get('message', ''))
        current_step = kwargs.get('current_step', 1)
        total_steps = kwargs.get('total_steps', 5)
        
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = min(100, max(0, step_progress))
            ui_components['current_progress'].description = f"Step {current_step}/{total_steps}"
            if hasattr(ui_components['current_progress'], 'layout'):
                ui_components['current_progress'].layout.visibility = 'visible'
        
        if 'step_label' in ui_components:
            ui_components['step_label'].value = step_message
            if hasattr(ui_components['step_label'], 'layout'):
                ui_components['step_label'].layout.visibility = 'visible'
    
    # Register observers untuk berbagai event
    progress_events = [
        'DOWNLOAD_PROGRESS', 'EXPORT_PROGRESS', 'BACKUP_PROGRESS', 
        'ZIP_PROCESSING_PROGRESS', 'PULL_DATASET_PROGRESS'
    ]
    
    step_events = [
        'DOWNLOAD_STEP_PROGRESS', 'EXPORT_STEP_PROGRESS', 'BACKUP_STEP_PROGRESS'
    ]
    
    observer_manager.create_simple_observer(progress_events, overall_progress_observer, name="overall_progress")
    observer_manager.create_simple_observer(step_events, step_progress_observer, name="step_progress")
    
    # Simpan observer manager
    ui_components['observer_manager'] = observer_manager
    ui_components['progress_setup'] = True
    
    return ui_components

"""
File: smartcash/ui/dataset/download/utils/progress_updater.py
Deskripsi: Utilitas untuk update progress yang konsisten dengan service layer
"""

def show_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan progress container dan set pesan awal."""
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    update_progress(ui_components, 0, message)

def update_progress(ui_components: Dict[str, Any], value: int, message: str = None) -> None:
    """Update progress bar dengan notifikasi ke observer."""
    value = max(0, min(100, value))
    
    # Update UI langsung
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = value
        ui_components['progress_bar'].description = f"Progress: {value}%"
        ui_components['progress_bar'].layout.visibility = 'visible'
    
    if message and 'overall_label' in ui_components:
        ui_components['overall_label'].value = message
        ui_components['overall_label'].layout.visibility = 'visible'
    
    # Notifikasi via observer untuk service layer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            'DOWNLOAD_PROGRESS', 
            ui_components, 
            progress=value, 
            message=message or f"Progress: {value}%"
        )

def update_step_progress(ui_components: Dict[str, Any], step: int, total_steps: int, 
                        step_progress: int, step_message: str) -> None:
    """Update step progress."""
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = max(0, min(100, step_progress))
        ui_components['current_progress'].description = f"Step {step}/{total_steps}"
        ui_components['current_progress'].layout.visibility = 'visible'
    
    if 'step_label' in ui_components:
        ui_components['step_label'].value = step_message
        ui_components['step_label'].layout.visibility = 'visible'
    
    # Notifikasi via observer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            'DOWNLOAD_STEP_PROGRESS',
            ui_components,
            current_step=step,
            total_steps=total_steps,
            step_progress=step_progress,
            step_message=step_message
        )

def reset_progress(ui_components: Dict[str, Any]) -> None:
    """Reset semua progress indicator."""
    for widget_key in ['progress_bar', 'current_progress']:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'hidden'
    
    for label_key in ['overall_label', 'step_label']:
        if label_key in ui_components:
            ui_components[label_key].value = ""
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'hidden'
    
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'none'

"""
File: smartcash/ui/dataset/download/utils/progress_updater.py
Deskripsi: Utilitas untuk update progress yang konsisten dengan service layer
"""

from typing import Dict, Any

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
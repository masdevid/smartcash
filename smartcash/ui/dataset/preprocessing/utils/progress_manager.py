"""
File: smartcash/ui/dataset/preprocessing/utils/progress_manager.py
Deskripsi: Manager untuk progress tracking pada preprocessing dataset
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    PreprocessingUIEvents, 
    notify_progress, 
    notify_step_progress
)

def setup_multi_progress(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup multi-progress tracking untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Key untuk progress tracking
    progress_keys = {
        'progress_bar': 'progress_bar',
        'overall_label': 'overall_label',
        'current_progress': 'current_progress',
        'step_label': 'step_label'
    }
    
    # Cek apakah semua komponen progress tracking tersedia
    has_progress_components = all(key in ui_components for key in progress_keys.values())
    
    if not has_progress_components:
        # Log jika komponen tidak lengkap
        log_message(ui_components, "Beberapa komponen progress tracking tidak tersedia", "warning", "⚠️")
    
    # Setup progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
        ui_components['progress_bar'].layout.visibility = 'hidden'
        ui_components['progress_bar'].value = 0
    
    # Setup current progress
    if 'current_progress' in ui_components and hasattr(ui_components['current_progress'], 'layout'):
        ui_components['current_progress'].layout.visibility = 'hidden'
        ui_components['current_progress'].value = 0
    
    # Setup overall label
    if 'overall_label' in ui_components and hasattr(ui_components['overall_label'], 'layout'):
        ui_components['overall_label'].layout.visibility = 'hidden'
        ui_components['overall_label'].value = ""
    
    # Setup step label
    if 'step_label' in ui_components and hasattr(ui_components['step_label'], 'layout'):
        ui_components['step_label'].layout.visibility = 'hidden'
        ui_components['step_label'].value = ""
    
    # Setup helper functions untuk update progress
    ui_components['update_progress'] = lambda progress, total=100, message="": update_progress(ui_components, progress, total, message)
    ui_components['update_step_progress'] = lambda step_progress, step_total=100, step_message="", current_step=1, total_steps=1: update_step_progress(ui_components, step_progress, step_total, step_message, current_step, total_steps)
    ui_components['reset_progress'] = lambda: reset_progress_bar(ui_components)
    
    # Log setup berhasil
    log_message(ui_components, "Progress tracking berhasil disetup", "debug", "✅")
    
    return ui_components

def setup_progress_indicator(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup indikator progress untuk preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup progress container jika tersedia
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'none'
        ui_components['progress_container'].layout.visibility = 'hidden'
    
    return ui_components

def update_progress(ui_components: Dict[str, Any], progress: int, total: int = 100, message: str = "") -> None:
    """
    Update progress bar dan label.
    
    Args:
        ui_components: Dictionary komponen UI
        progress: Nilai progress saat ini
        total: Total nilai progress
        message: Pesan progress
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Pastikan progress adalah integer
    try:
        progress = int(float(progress))
    except (ValueError, TypeError):
        progress = 0
    
    # Pastikan progress berada dalam range valid
    progress = max(0, min(progress, 100))
    
    # Update progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        # Show progress bar jika tersembunyi
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
        
        # Update nilai progress bar
        ui_components['progress_bar'].value = progress
        ui_components['progress_bar'].description = f"Progress: {progress}%"
    
    # Update label jika ada pesan
    if message and 'overall_label' in ui_components and hasattr(ui_components['overall_label'], 'value'):
        # Show label jika tersembunyi
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'visible'
        
        # Update nilai label
        ui_components['overall_label'].value = message
    
    # Notifikasi progress melalui observer
    notify_progress(ui_components, progress, total, message)

def update_step_progress(ui_components: Dict[str, Any], step_progress: int, step_total: int = 100, 
                        step_message: str = "", current_step: int = 1, total_steps: int = 1) -> None:
    """
    Update step progress bar dan label.
    
    Args:
        ui_components: Dictionary komponen UI
        step_progress: Nilai progress step saat ini
        step_total: Total nilai progress step
        step_message: Pesan step progress
        current_step: Langkah saat ini
        total_steps: Total langkah
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Pastikan step_progress adalah integer
    try:
        step_progress = int(float(step_progress))
    except (ValueError, TypeError):
        step_progress = 0
    
    # Pastikan step_progress berada dalam range valid
    step_progress = max(0, min(step_progress, 100))
    
    # Update step progress bar
    if 'current_progress' in ui_components and hasattr(ui_components['current_progress'], 'value'):
        # Show progress bar jika tersembunyi
        if hasattr(ui_components['current_progress'], 'layout'):
            ui_components['current_progress'].layout.visibility = 'visible'
        
        # Update nilai progress bar
        ui_components['current_progress'].value = step_progress
        ui_components['current_progress'].description = f"Step {current_step}/{total_steps}"
    
    # Update label jika ada pesan
    if step_message and 'step_label' in ui_components and hasattr(ui_components['step_label'], 'value'):
        # Show label jika tersembunyi
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'visible'
        
        # Update nilai label
        ui_components['step_label'].value = step_message
    
    # Notifikasi step progress melalui observer
    notify_step_progress(ui_components, step_progress, step_total, step_message, current_step, total_steps)

def reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar dan label.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Reset progress bar
    if 'progress_bar' in ui_components:
        # Hide progress bar
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'hidden'
        
        # Reset nilai progress bar
        ui_components['progress_bar'].value = 0
    
    # Reset overall label
    if 'overall_label' in ui_components:
        # Hide label
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'hidden'
        
        # Reset nilai label
        ui_components['overall_label'].value = ""
    
    # Reset current progress
    if 'current_progress' in ui_components:
        # Hide progress bar
        if hasattr(ui_components['current_progress'], 'layout'):
            ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset nilai progress bar
        ui_components['current_progress'].value = 0
    
    # Reset step label
    if 'step_label' in ui_components:
        # Hide label
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'hidden'
        
        # Reset nilai label
        ui_components['step_label'].value = ""
    
    # Reset progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.visibility = 'hidden'
        ui_components['progress_container'].layout.display = 'none'

def start_progress(ui_components: Dict[str, Any], message: str = "Memulai preprocessing...") -> None:
    """
    Memulai progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan awal progress
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Reset progress dulu
    reset_progress_bar(ui_components)
    
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.visibility = 'visible'
        ui_components['progress_container'].layout.display = 'block'
    
    # Update progress awal
    update_progress(ui_components, 0, 100, message)
    
    # Notifikasi start melalui observer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            PreprocessingUIEvents.PROGRESS_START,
            ui_components,
            progress=0,
            total=100,
            message=message
        )

def complete_progress(ui_components: Dict[str, Any], message: str = "Preprocessing selesai") -> None:
    """
    Menyelesaikan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan akhir progress
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Update progress ke 100%
    update_progress(ui_components, 100, 100, message)
    
    # Notifikasi complete melalui observer
    if 'observer_manager' in ui_components:
        ui_components['observer_manager'].notify(
            PreprocessingUIEvents.PROGRESS_COMPLETE,
            ui_components,
            progress=100,
            total=100,
            message=message
        ) 
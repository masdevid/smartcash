"""
File: smartcash/ui/pretrained_model/utils/progress.py
Deskripsi: Utilitas untuk progress tracking pada UI model
"""

from typing import Dict, Any

def update_progress_ui(ui_components: Dict[str, Any], progress: int, total: int, message: str) -> None:
    """
    Update progress bar dan label dengan progress terbaru
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan progress_bar dan progress_label
        progress: Nilai progress saat ini
        total: Nilai total progress
        message: Pesan yang ditampilkan
    """
    # Gunakan API update_progress jika tersedia (API baru)
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress'](progress/total*100, message)
        return
    
    # Fallback ke API update_progress_bar jika tersedia (API lama)
    if 'update_progress_bar' in ui_components and callable(ui_components['update_progress_bar']):
        ui_components['update_progress_bar'](progress, total, message)
        return

    # Fallback ke akses langsung komponen jika tersedia
    # Cek jika progress_bar adalah ProgressTracker (API baru)
    if 'progress_bar' in ui_components:
        if hasattr(ui_components['progress_bar'], 'update'):
            # ProgressTracker object
            ui_components['progress_bar'].update(progress/total*100, message)
        elif hasattr(ui_components['progress_bar'], 'value'):
            # Widget langsung
            ui_components['progress_bar'].value = progress
            ui_components['progress_bar'].max = total
            
            # Update label jika tersedia
            if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
                ui_components['progress_label'].value = f"{progress}/{total}: {message}"

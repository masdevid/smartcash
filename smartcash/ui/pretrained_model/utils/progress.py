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
    # Update dengan fungsi khusus jika tersedia
    if 'update_progress_bar' in ui_components and callable(ui_components['update_progress_bar']):
        ui_components['update_progress_bar'](progress, total, message)
        return

    # Atau update langsung komponen jika tersedia
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = progress
        ui_components['progress_bar'].max = total
    
    if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
        ui_components['progress_label'].value = f"{progress}/{total}: {message}"

"""
File: smartcash/ui/setup/env_config/components/progress_tracking.py
Deskripsi: Utilitas untuk progress tracking dengan reset capability - diperbaiki untuk error handling
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def reset_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """
    Reset progress bar ke 0 dengan optional error message
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan opsional untuk ditampilkan
    """
    # Reset progress bar ke 0
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].description = "0%"
    
    # Update message jika ada
    if 'progress_message' in ui_components:
        if message:
            ui_components['progress_message'].value = message
        else:
            ui_components['progress_message'].value = ""
    
    # Reset progress container style untuk error
    if 'progress_container' in ui_components and message:
        # Add error styling jika ada message
        ui_components['progress_container'].layout.border = '1px solid #dc3545'

def update_progress(ui_components: Dict[str, Any], current: int, total: int, message: str = "") -> None:
    """
    Update progress bar dengan nilai dan pesan baru
    
    Args:
        ui_components: Dictionary komponen UI
        current: Nilai progress saat ini
        total: Total maksimum progress
        message: Pesan progress
    """
    if 'progress_bar' in ui_components:
        percentage = min(100, max(0, int((current / total) * 100)))
        ui_components['progress_bar'].value = percentage
        ui_components['progress_bar'].description = f"{percentage}%"
        
        # Reset error styling jika progress normal
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.border = '1px solid #ddd'
    
    if 'progress_message' in ui_components and message:
        ui_components['progress_message'].value = message

def create_progress_tracking(module_name: str = "progress", width: str = "100%") -> Dict[str, widgets.Widget]:
    """
    Buat komponen progress tracking dengan reset capability
    
    Args:
        module_name: Nama modul untuk tracking
        width: Lebar progress bar
        
    Returns:
        Dictionary komponen progress
    """
    # Progress bar dengan styling yang lebih baik
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width=width, margin='5px 0')
    )
    
    # Progress message
    progress_message = widgets.Label(
        value="",
        layout=widgets.Layout(
            width=width,
            margin='2px 0',
            overflow='hidden'
        ),
        style={'font_size': '12px', 'text_color': '#666'}
    )
    
    # Container dengan proper styling
    progress_container = widgets.VBox(
        [progress_bar, progress_message],
        layout=widgets.Layout(
            width=width,
            border='1px solid #ddd',
            border_radius='4px',
            padding='8px',
            margin='5px 0'
        )
    )
    
    return {
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'progress_container': progress_container
    }
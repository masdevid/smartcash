"""
File: smartcash/ui/setup/env_config/components/progress_tracking.py
Deskripsi: Progress tracking dengan visibility control yang proper
"""

import ipywidgets as widgets
from typing import Dict, Any

def reset_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """Reset progress bar ke 0"""
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].description = "0%"
    
    if 'progress_message' in ui_components:
        ui_components['progress_message'].value = message or ""
    
    # Show progress container pada reset
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.visibility = 'visible'

def update_progress(ui_components: Dict[str, Any], current: int, total: int, message: str = "") -> None:
    """Update progress bar dengan nilai baru"""
    if 'progress_bar' in ui_components:
        percentage = min(100, max(0, int((current / total) * 100)))
        ui_components['progress_bar'].value = percentage
        ui_components['progress_bar'].description = f"{percentage}%"
    
    if 'progress_message' in ui_components and message:
        ui_components['progress_message'].value = message

def hide_progress(ui_components: Dict[str, Any]) -> None:
    """Hide progress container setelah selesai"""
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.visibility = 'hidden'

def create_progress_tracking(module_name: str = "progress", width: str = "100%") -> Dict[str, widgets.Widget]:
    """Create progress tracking components"""
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        layout=widgets.Layout(width=width, margin='5px 0')
    )
    
    progress_message = widgets.Label(
        value="",
        layout=widgets.Layout(width=width, margin='2px 0'),
        style={'font_size': '12px', 'text_color': '#666'}
    )
    
    return {
        'progress_bar': progress_bar,
        'progress_message': progress_message
    }
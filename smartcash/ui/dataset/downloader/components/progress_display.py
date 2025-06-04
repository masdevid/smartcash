"""
File: smartcash/ui/dataset/downloader/components/progress_display.py
Deskripsi: Progress display component dengan step-by-step tracking dan overall progress
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS

def create_progress_display() -> Dict[str, Any]:
    """Create progress display dengan step tracking dan overall progress."""
    
    # Overall progress bar
    overall_progress = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Overall:',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': '80px', 'bar_color': COLORS.get('primary', '#007bff')}
    )
    
    # Step progress bar  
    step_progress = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Step:',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': '80px', 'bar_color': COLORS.get('success', '#28a745')}
    )
    
    # Status message
    status_message = widgets.HTML(
        value="<div style='padding: 8px; color: #495057; font-size: 14px;'>Siap untuk memulai download</div>",
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Container
    container = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0; color: #333;'>📊 Progress Download</h4>"),
        overall_progress,
        step_progress,
        status_message
    ], layout=widgets.Layout(
        width='100%', 
        padding='15px',
        border=f'1px solid {COLORS.get("border", "#ddd")}',
        border_radius='6px',
        background_color='#f8f9fa',
        margin='10px 0',
        display='none'  # Hidden by default
    ))
    
    return {
        'container': container,
        'overall_progress': overall_progress,
        'step_progress': step_progress,
        'status_message': status_message
    }

def update_overall_progress(components: Dict[str, Any], progress: int, message: str = "") -> None:
    """Update overall progress bar dengan one-liner."""
    (setattr(components['overall_progress'], 'value', max(0, min(100, progress))),
     setattr(components['status_message'], 'value', 
             f"<div style='padding: 8px; color: #495057; font-size: 14px;'>{message}</div>") if message else None)

def update_step_progress(components: Dict[str, Any], progress: int, step_name: str = "") -> None:
    """Update step progress bar dengan one-liner."""
    (setattr(components['step_progress'], 'value', max(0, min(100, progress))),
     setattr(components['step_progress'], 'description', f'{step_name}:' if step_name else 'Step:'))

def show_progress(components: Dict[str, Any], message: str = "Memulai proses download") -> None:
    """Show progress container dengan one-liner."""
    (setattr(components['container'].layout, 'display', 'flex'),
     update_overall_progress(components, 0, message),
     update_step_progress(components, 0))

def hide_progress(components: Dict[str, Any]) -> None:
    """Hide progress container dengan one-liner."""
    setattr(components['container'].layout, 'display', 'none')

def complete_progress(components: Dict[str, Any], message: str = "Download selesai!") -> None:
    """Complete progress dengan success state."""
    (update_overall_progress(components, 100, f"✅ {message}"),
     update_step_progress(components, 100, "Selesai"),
     setattr(components['overall_progress'].style, 'bar_color', COLORS.get('success', '#28a745')))

def error_progress(components: Dict[str, Any], message: str = "Terjadi error") -> None:
    """Set error state untuk progress."""
    (update_overall_progress(components, 0, f"❌ {message}"),
     update_step_progress(components, 0, "Error"),
     setattr(components['overall_progress'].style, 'bar_color', COLORS.get('danger', '#dc3545')))
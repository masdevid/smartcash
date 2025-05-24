"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Fixed progress tracking dengan overall dan step progress yang konsisten
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional


def create_progress_tracking(
    module_name: str = "processing",
    show_step_progress: bool = True,
    show_overall_progress: bool = True,
    show_current_progress: bool = True,
    width: str = '100%'
) -> Dict[str, Any]:
    """
    Buat komponen progress tracking dengan 3-level hierarchy yang konsisten.
    
    Args:
        module_name: Nama modul untuk identifikasi
        show_step_progress: Tampilkan step progress (preprocessing steps)
        show_overall_progress: Tampilkan overall progress (keseluruhan proses)
        show_current_progress: Tampilkan current progress (per split/batch)
        width: Lebar container
        
    Returns:
        Dict komponen progress tracking
    """
    
    components = []
    
    # === OVERALL PROGRESS (Keseluruhan proses) ===
    overall_progress = None
    overall_label = None
    
    if show_overall_progress:
        overall_label = widgets.HTML(
            value="<div style='margin-bottom: 5px; color: #495057; font-weight: bold;'>📊 Overall Progress: Siap memulai</div>",
            layout=widgets.Layout(width=width)
        )
        
        overall_progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            bar_style='info',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width=width, height='25px')
        )
        
        components.extend([overall_label, overall_progress])
    
    # === STEP PROGRESS (Per tahap: train, valid, test) ===
    step_progress = None
    step_label = None
    
    if show_step_progress:
        step_label = widgets.HTML(
            value="<div style='margin: 10px 0 5px 0; color: #6c757d; font-size: 14px;'>🔄 Step: Menunggu</div>",
            layout=widgets.Layout(width=width)
        )
        
        step_progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            bar_style='',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width=width, height='20px')
        )
        
        components.extend([step_label, step_progress])
    
    # === CURRENT PROGRESS (Per file/batch dalam step) ===
    current_progress = None
    current_label = None
    
    if show_current_progress:
        current_label = widgets.HTML(
            value="<div style='margin: 8px 0 5px 0; color: #868e96; font-size: 12px;'>⚡ Current: -</div>",
            layout=widgets.Layout(width=width)
        )
        
        current_progress = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            bar_style='success',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width=width, height='15px')
        )
        
        components.extend([current_label, current_progress])
    
    # Container dengan visibility control
    progress_container = widgets.VBox(
        components,
        layout=widgets.Layout(
            width=width,
            padding='15px',
            margin='10px 0',
            border='1px solid #dee2e6',
            border_radius='5px',
            background_color='#f8f9fa',
            visibility='hidden',  # Hidden by default
            display='none'
        )
    )
    
    return {
        # Main container
        'progress_container': progress_container,
        
        # Overall progress (keseluruhan proses)
        'overall_progress': overall_progress,
        'overall_label': overall_label,
        'progress_bar': overall_progress,  # Alias untuk backward compatibility
        
        # Step progress (per split: train, valid, test)
        'step_progress': step_progress,
        'step_label': step_label,
        
        # Current progress (per file dalam split)
        'current_progress': current_progress,
        'current_label': current_label,
        
        # Helper methods
        'show_container': lambda: _show_progress_container(progress_container),
        'hide_container': lambda: _hide_progress_container(progress_container),
        'reset_all': lambda: _reset_all_progress(overall_progress, step_progress, current_progress,
                                               overall_label, step_label, current_label),
        
        # Module info
        'module_name': module_name
    }


def _show_progress_container(container):
    """Show progress container."""
    container.layout.visibility = 'visible'
    container.layout.display = 'block'


def _hide_progress_container(container):
    """Hide progress container."""
    container.layout.visibility = 'hidden'
    container.layout.display = 'none'


def _reset_all_progress(overall_progress, step_progress, current_progress,
                       overall_label, step_label, current_label):
    """Reset semua progress indicators."""
    if overall_progress:
        overall_progress.value = 0
        overall_progress.bar_style = 'info'
    
    if step_progress:
        step_progress.value = 0
        step_progress.bar_style = ''
    
    if current_progress:
        current_progress.value = 0
        current_progress.bar_style = 'success'
    
    if overall_label:
        overall_label.value = "<div style='margin-bottom: 5px; color: #495057; font-weight: bold;'>📊 Overall Progress: Siap memulai</div>"
    
    if step_label:
        step_label.value = "<div style='margin: 10px 0 5px 0; color: #6c757d; font-size: 14px;'>🔄 Step: Menunggu</div>"
    
    if current_label:
        current_label.value = "<div style='margin: 8px 0 5px 0; color: #868e96; font-size: 12px;'>⚡ Current: -</div>"


def update_overall_progress(components: Dict[str, Any], progress: int, total: int, message: str = ""):
    """Update overall progress (keseluruhan proses)."""
    if 'overall_progress' in components and components['overall_progress']:
        percentage = min((progress / max(total, 1)) * 100, 100)
        components['overall_progress'].value = percentage
        
        # Update bar style based on progress
        if percentage >= 100:
            components['overall_progress'].bar_style = 'success'
        elif percentage >= 50:
            components['overall_progress'].bar_style = 'info'
        else:
            components['overall_progress'].bar_style = ''
    
    if 'overall_label' in components and components['overall_label'] and message:
        components['overall_label'].value = f"<div style='margin-bottom: 5px; color: #495057; font-weight: bold;'>📊 Overall: {message} ({progress}/{total})</div>"


def update_step_progress(components: Dict[str, Any], step: int, total_steps: int, step_name: str = ""):
    """Update step progress (per split)."""
    if 'step_progress' in components and components['step_progress']:
        percentage = min((step / max(total_steps, 1)) * 100, 100)
        components['step_progress'].value = percentage
        
        # Update bar style
        if percentage >= 100:
            components['step_progress'].bar_style = 'success'
        else:
            components['step_progress'].bar_style = 'info'
    
    if 'step_label' in components and components['step_label']:
        step_text = f"Step {step}/{total_steps}"
        if step_name:
            step_text += f": {step_name}"
        components['step_label'].value = f"<div style='margin: 10px 0 5px 0; color: #6c757d; font-size: 14px;'>🔄 {step_text}</div>"


def update_current_progress(components: Dict[str, Any], current: int, total: int, message: str = ""):
    """Update current progress (per file/batch)."""
    if 'current_progress' in components and components['current_progress']:
        percentage = min((current / max(total, 1)) * 100, 100)
        components['current_progress'].value = percentage
    
    if 'current_label' in components and components['current_label']:
        if message:
            components['current_label'].value = f"<div style='margin: 8px 0 5px 0; color: #868e96; font-size: 12px;'>⚡ {message} ({current}/{total})</div>"
        else:
            components['current_label'].value = f"<div style='margin: 8px 0 5px 0; color: #868e96; font-size: 12px;'>⚡ Progress: {current}/{total}</div>"
"""
File: smartcash/ui/dataset/augmentation/components/progress_component.py
Deskripsi: Komponen tracking progress untuk augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS

def create_progress_component() -> Dict[str, Any]:
    """
    Buat komponen tracking progress untuk augmentasi dataset.
    
    Returns:
        Dictionary berisi komponen progress tracking
    """
    # Progress bars untuk tracking proses
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Overall:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Current:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    # Label untuk progress messages
    overall_message = widgets.HTML(
        value="",
        layout=widgets.Layout(visibility='hidden')
    )
    
    step_message = widgets.HTML(
        value="",
        layout=widgets.Layout(visibility='hidden')
    )
    
    # Container untuk komponen progress
    progress_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
        progress_bar, 
        current_progress,
        overall_message,
        step_message
    ])
    
    return {
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'overall_message': overall_message,
        'step_message': step_message,
        'container': progress_container
    }
"""
File: smartcash/ui/dataset/augmentation/components/output_component.py
Deskripsi: Komponen output untuk augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET

def create_output_component() -> Dict[str, Any]:
    """
    Buat komponen output untuk augmentasi dataset.
    
    Returns:
        Dictionary berisi komponen output
    """
    # Status output untuk log
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Log accordion untuk menampilkan status
    log_accordion = widgets.Accordion(children=[status], selected_index=None)
    log_accordion.set_title(0, f"{ICONS['file']} Augmentation Logs")
    
    # Visualization container
    visualization_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            min_height='100px',
            display='none'
        )
    )
    
    # Summary container
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            display='none'
        )
    )
    
    return {
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'visualization_container': visualization_container
    }
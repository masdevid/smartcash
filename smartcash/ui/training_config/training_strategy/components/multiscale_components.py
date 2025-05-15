"""
File: smartcash/ui/training_config/training_strategy/components/multiscale_components.py
Deskripsi: Komponen multi-scale untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS

def create_training_strategy_multiscale_components() -> Dict[str, Any]:
    """
    Membuat komponen UI multi-scale untuk strategi pelatihan.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Parameter multi-scale
    ui_components['multi_scale'] = widgets.Checkbox(
        value=True,
        description='Enable multi-scale training',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat box untuk parameter multi-scale
    ui_components['multiscale_box'] = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('scale', '📏')} Parameter Multi-scale</h4>"),
        ui_components['multi_scale']
    ], layout=widgets.Layout(
        width='auto',
        padding='10px',
        border='1px solid #ddd',
        border_radius='5px',
        overflow='visible'
    ))
    
    return ui_components

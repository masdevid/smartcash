"""
File: smartcash/ui/training_config/training_strategy/components/validation_components.py
Deskripsi: Komponen validasi untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS

def create_training_strategy_validation_components() -> Dict[str, Any]:
    """
    Membuat komponen UI validasi untuk strategi pelatihan.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Parameter validation frequency
    ui_components['validation_frequency'] = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        step=1,
        description='Validation frequency:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter IoU threshold
    ui_components['iou_threshold'] = widgets.FloatSlider(
        value=0.6,
        min=0.1,
        max=0.9,
        step=0.05,
        description='IoU threshold:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter confidence threshold
    ui_components['conf_threshold'] = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.0001,
        description='Conf threshold:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat box untuk parameter validasi
    ui_components['validation_box'] = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('check', '✓')} Parameter Validasi</h4>"),
        ui_components['validation_frequency'],
        ui_components['iou_threshold'],
        ui_components['conf_threshold']
    ], layout=widgets.Layout(
        width='auto',
        padding='10px',
        border='1px solid #ddd',
        border_radius='5px',
        overflow='visible'
    ))
    
    return ui_components

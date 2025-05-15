"""
File: smartcash/ui/training_config/hyperparameters/components/basic_components.py
Deskripsi: Komponen UI dasar untuk konfigurasi hyperparameter
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS

def create_hyperparameters_basic_components() -> Dict[str, Any]:
    """
    Membuat komponen UI dasar untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Parameter batch size
    ui_components['batch_size_slider'] = widgets.IntSlider(
        value=16,
        min=1,
        max=128,
        step=1,
        description='Batch Size:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter image size
    ui_components['image_size_slider'] = widgets.IntSlider(
        value=640,
        min=320,
        max=1280,
        step=32,
        description='Image Size:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter epochs
    ui_components['epochs_slider'] = widgets.IntSlider(
        value=100,
        min=1,
        max=500,
        step=1,
        description='Epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat box untuk parameter dasar
    ui_components['basic_box'] = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('settings', '⚙️')} Parameter Dasar</h4>"),
        ui_components['batch_size_slider'],
        ui_components['image_size_slider'],
        ui_components['epochs_slider']
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        border='1px solid #ddd',
        border_radius='5px',
        height='100%'
    ))
    
    return ui_components

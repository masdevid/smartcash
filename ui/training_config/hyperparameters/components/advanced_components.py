"""
File: smartcash/ui/training_config/hyperparameters/components/advanced_components.py
Deskripsi: Komponen UI lanjutan untuk konfigurasi hyperparameter
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS

def create_hyperparameters_advanced_components() -> Dict[str, Any]:
    """
    Membuat komponen UI lanjutan untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Parameter augmentasi
    ui_components['augment_checkbox'] = widgets.Checkbox(
        value=True,
        description='Gunakan Augmentasi',
        style={'description_width': 'auto'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter dropout
    ui_components['dropout_slider'] = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.5,
        step=0.01,
        description='Dropout:',
        readout_format='.2f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter box loss gain
    ui_components['box_loss_gain_slider'] = widgets.FloatSlider(
        value=0.05,
        min=0.01,
        max=0.1,
        step=0.01,
        description='Box Loss Gain:',
        readout_format='.2f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter class loss gain
    ui_components['cls_loss_gain_slider'] = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=1.0,
        step=0.1,
        description='Class Loss Gain:',
        readout_format='.1f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter object loss gain
    ui_components['obj_loss_gain_slider'] = widgets.FloatSlider(
        value=1.0,
        min=0.5,
        max=2.0,
        step=0.1,
        description='Object Loss Gain:',
        readout_format='.1f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter early stopping
    ui_components['early_stopping_checkbox'] = widgets.Checkbox(
        value=True,
        description='Early Stopping',
        style={'description_width': 'auto'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter patience
    ui_components['patience_slider'] = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Patience:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter min delta
    ui_components['min_delta_slider'] = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.0001,
        description='Min Delta:',
        readout_format='.4f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat box untuk parameter lanjutan
    ui_components['advanced_box'] = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('advanced', '🔧')} Parameter Lanjutan</h4>"),
        ui_components['augment_checkbox'],
        ui_components['dropout_slider'],
        widgets.HTML("<hr style='margin: 10px 0'>"),
        widgets.HTML("<b>Parameter Loss</b>"),
        ui_components['box_loss_gain_slider'],
        ui_components['cls_loss_gain_slider'],
        ui_components['obj_loss_gain_slider'],
        widgets.HTML("<hr style='margin: 10px 0'>"),
        widgets.HTML("<b>Early Stopping</b>"),
        ui_components['early_stopping_checkbox'],
        ui_components['patience_slider'],
        ui_components['min_delta_slider']
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        border='1px solid #ddd',
        border_radius='5px',
        height='100%'
    ))
    
    return ui_components

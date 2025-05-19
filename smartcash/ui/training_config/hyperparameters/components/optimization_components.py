"""
File: smartcash/ui/training_config/hyperparameters/components/optimization_components.py
Deskripsi: Komponen UI optimasi untuk konfigurasi hyperparameter
"""

from typing import Dict, Any
import ipywidgets as widgets
import numpy as np

from smartcash.ui.utils.constants import ICONS

def create_hyperparameters_optimization_components() -> Dict[str, Any]:
    """
    Membuat komponen UI optimasi untuk hyperparameter.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Parameter optimizer
    ui_components['optimizer_dropdown'] = widgets.Dropdown(
        options=['SGD', 'Adam', 'AdamW', 'RMSprop'],
        value='SGD',
        description='Optimizer:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter learning rate
    ui_components['learning_rate_slider'] = widgets.FloatLogSlider(
        value=0.01,
        base=10,
        min=-5,  # 10^-5 = 0.00001
        max=-1,  # 10^-1 = 0.1
        step=0.1,
        description='Learning Rate:',
        readout_format='.6f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter weight decay
    ui_components['weight_decay_slider'] = widgets.FloatLogSlider(
        value=0.0005,
        base=10,
        min=-6,  # 10^-6 = 0.000001
        max=-2,  # 10^-2 = 0.01
        step=0.1,
        description='Weight Decay:',
        readout_format='.6f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter momentum
    ui_components['momentum_slider'] = widgets.FloatSlider(
        value=0.937,
        min=0.8,
        max=0.999,
        step=0.001,
        description='Momentum:',
        readout_format='.3f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter scheduler
    ui_components['scheduler_dropdown'] = widgets.Dropdown(
        options=['cosine', 'linear', 'step', 'exp', 'none'],
        value='cosine',
        description='LR Scheduler:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter warmup epochs
    ui_components['warmup_epochs_slider'] = widgets.IntSlider(
        value=3,
        min=0,
        max=10,
        step=1,
        description='Warmup Epochs:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter warmup momentum
    ui_components['warmup_momentum_slider'] = widgets.FloatSlider(
        value=0.8,
        min=0.0,
        max=0.999,
        step=0.001,
        description='Warmup Momentum:',
        readout_format='.3f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Parameter warmup bias lr
    ui_components['warmup_bias_lr_slider'] = widgets.FloatSlider(
        value=0.1,
        min=0.0,
        max=0.5,
        step=0.01,
        description='Warmup Bias LR:',
        readout_format='.2f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    # Buat box untuk parameter optimasi
    ui_components['optimization_box'] = widgets.VBox([
        widgets.HTML(f"<h4>{ICONS.get('optimization', '🔄')} Parameter Optimasi</h4>"),
        ui_components['optimizer_dropdown'],
        ui_components['learning_rate_slider'],
        ui_components['weight_decay_slider'],
        ui_components['momentum_slider'],
        widgets.HTML("<hr style='margin: 10px 0'>"),
        ui_components['scheduler_dropdown'],
        ui_components['warmup_epochs_slider'],
        ui_components['warmup_momentum_slider'],
        ui_components['warmup_bias_lr_slider']
    ], layout=widgets.Layout(
        width='100%',
        padding='5px',
        border='1px solid #ddd',
        border_radius='5px',
        height='100%',
        overflow='visible'
    ))
    
    return ui_components

"""
File: smartcash/ui/training_config/training_strategy/components/utils_components.py
Deskripsi: Komponen utilitas untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS

def create_training_strategy_utils_components() -> Dict[str, Any]:
    """
    Membuat komponen UI utilitas untuk strategi pelatihan.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Parameter experiment name
    ui_components['experiment_name'] = widgets.Text(
        value='efficientnet_b4_training',
        description='Experiment name:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='500px')
    )
    
    # Parameter checkpoint directory
    ui_components['checkpoint_dir'] = widgets.Text(
        value='/content/runs/train/checkpoints',
        description='Checkpoint dir:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='500px')
    )
    
    # Parameter tensorboard
    ui_components['tensorboard'] = widgets.Checkbox(
        value=True,
        description='Enable TensorBoard',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    # Parameter log metrics
    ui_components['log_metrics_every'] = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description='Log metrics every:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Parameter visualize batch
    ui_components['visualize_batch_every'] = widgets.IntSlider(
        value=100,
        min=10,
        max=500,
        step=10,
        description='Visualize batch every:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Parameter gradient clipping
    ui_components['gradient_clipping'] = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description='Gradient clipping:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Parameter mixed precision
    ui_components['mixed_precision'] = widgets.Checkbox(
        value=True,
        description='Enable mixed precision',
        style={'description_width': '200px'},
        layout=widgets.Layout(width='300px')
    )
    
    # Parameter layer mode
    ui_components['layer_mode'] = widgets.RadioButtons(
        options=['single', 'multilayer'],
        value='single',
        description='Layer mode:',
        style={'description_width': '150px'},
        layout=widgets.Layout(width='400px')
    )
    
    # Buat box untuk parameter utilitas
    ui_components['utils_box'] = widgets.VBox([
        ui_components['experiment_name'],
        ui_components['checkpoint_dir'],
        ui_components['tensorboard'],
        ui_components['log_metrics_every'],
        ui_components['visualize_batch_every'],
        ui_components['gradient_clipping'],
        ui_components['mixed_precision'],
        ui_components['layer_mode']
    ])
    
    return ui_components

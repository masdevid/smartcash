"""
File: smartcash/ui_components/hyperparameters.py
Author: Generated
Deskripsi: Komponen UI untuk konfigurasi hyperparameter model SmartCash.
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import create_component_header, create_info_box, create_section_title

def create_hyperparameters_ui():
    """
    Buat komponen UI untuk konfigurasi hyperparameter model.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Model Hyperparameters",
        "Konfigurasi parameter training untuk optimasi performa model",
        "🎛️"
    )
    
    # Basic hyperparameters section
    basic_section = create_section_title("Basic Hyperparameters", "📊")
    
    basic_params = widgets.VBox([
        widgets.IntSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description='Epochs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.IntSlider(
            value=16,
            min=4,
            max=64,
            step=4,
            description='Batch Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatLogSlider(
            value=0.01,
            base=10,
            min=-4,
            max=-1,
            step=0.1,
            description='Learning Rate:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Dropdown(
            options=['Adam', 'AdamW', 'SGD', 'RMSprop'],
            value='Adam',
            description='Optimizer:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    ])
    
    # Learning rate scheduler section
    scheduler_section = create_section_title("Learning Rate Scheduler", "📉")
    
    scheduler_params = widgets.VBox([
        widgets.Dropdown(
            options=['cosine', 'step', 'linear', 'exp', 'OneCycleLR', 'none'],
            value='cosine',
            description='Scheduler:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.FloatSlider(
            value=0.01,
            min=0.001,
            max=0.1,
            step=0.001,
            description='Final LR:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.IntSlider(
            value=5,
            min=1,
            max=10,
            description='Patience:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%',
            disabled=True)
        ),
        widgets.FloatSlider(
            value=0.1,
            min=0.01,
            max=0.5,
            step=0.05,
            description='Factor:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%',
            disabled=True)
        )
    ])
    
    # Early stopping section
    early_stopping_section = create_section_title("Early Stopping & Checkpointing", "🛑")
    
    early_stopping_params = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Enable early stopping',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            description='Patience:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Dropdown(
            options=['val_loss', 'val_mAP', 'val_f1', 'val_precision', 'val_recall'],
            value='val_mAP',
            description='Monitor:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.Checkbox(
            value=True,
            description='Save best model only',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=5,
            min=1,
            max=10,
            description='Save every N epochs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Advanced params section
    advanced_section = create_section_title("Advanced Parameters", "⚙️")
    
    advanced_params = widgets.VBox([
        widgets.FloatSlider(
            value=0.937,
            min=0.8,
            max=0.999,
            step=0.001,
            description='Momentum:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatLogSlider(
            value=0.0005,
            base=10,
            min=-5,
            max=-2,
            description='Weight Decay:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=False,
            description='Use EMA (Exponential Moving Average)',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=False,
            description='Use SWA (Stochastic Weight Averaging)',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Use mixed precision training',
            style={'description_width': 'initial'}
        )
    ])
    
    # Loss weights section
    loss_section = create_section_title("Loss Weights", "⚖️")
    
    loss_params = widgets.VBox([
        widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=0.2,
            step=0.01,
            description='Box Loss:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatSlider(
            value=0.5,
            min=0.1,
            max=1.0,
            step=0.05,
            description='Object Loss:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatSlider(
            value=0.5,
            min=0.1,
            max=1.0,
            step=0.05,
            description='Class Loss:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Tabs for advanced settings
    advanced_tabs = widgets.Tab()
    advanced_tabs.children = [
        scheduler_params,
        early_stopping_params,
        advanced_params,
        loss_params
    ]
    
    advanced_tabs.set_title(0, 'Scheduler')
    advanced_tabs.set_title(1, 'Early Stopping')
    advanced_tabs.set_title(2, 'Advanced')
    advanced_tabs.set_title(3, 'Loss Weights')
    
    # Action buttons
    save_button = widgets.Button(
        description='Save Configuration',
        button_style='primary',
        icon='save',
        layout=widgets.Layout(margin='0 10px 0 0')
    )
    
    reset_button = widgets.Button(
        description='Reset to Default',
        button_style='warning',
        icon='refresh',
        layout=widgets.Layout(margin='0 10px 0 0')
    )
    
    buttons_container = widgets.HBox([save_button, reset_button], 
                                     layout=widgets.Layout(margin='10px 0'))
    
    # Status output area
    status_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Help info
    help_info = create_info_box(
        "Tentang Hyperparameters",
        """
        <p>Parameter ini mengendalikan proses training model:</p>
        <ul>
            <li><strong>Epochs</strong>: Jumlah iterasi training pada dataset</li>
            <li><strong>Batch Size</strong>: Jumlah sampel yang diproses sebelum update weights</li>
            <li><strong>Learning Rate</strong>: Kecepatan penyesuaian weights model</li>
            <li><strong>Scheduler</strong>: Cara learning rate berubah selama training</li>
            <li><strong>Early Stopping</strong>: Menghentikan training ketika tidak ada peningkatan</li>
        </ul>
        <p><em>Tip: Parameter default biasanya bekerja dengan baik, sesuaikan dengan hati-hati!</em></p>
        """,
        'info'
    )
    
    # Visualization placeholder
    visualization_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='200px',
            max_height='300px',
            margin='10px 0',
            overflow='auto',
            display='none'
        )
    )
    
    # Assemble all components
    main_container.children = [
        header,
        basic_section,
        basic_params,
        advanced_section,
        advanced_tabs,
        buttons_container,
        status_output,
        visualization_output,
        help_info
    ]
    
    # Dictionary for access to components
    ui_components = {
        'ui': main_container,
        'basic_params': basic_params,
        'scheduler_params': scheduler_params,
        'early_stopping_params': early_stopping_params,
        'advanced_params': advanced_params,
        'loss_params': loss_params,
        'advanced_tabs': advanced_tabs,
        'save_button': save_button,
        'reset_button': reset_button,
        'status_output': status_output,
        'visualization_output': visualization_output
    }
    
    return ui_components
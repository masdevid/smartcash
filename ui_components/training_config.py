"""
File: smartcash/ui_components/training_config.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk konfigurasi training model SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from smartcash.utils.ui_utils import (
    create_header, create_section_title, create_info_alert,
    create_status_indicator, create_tab_view, create_info_box
)

def create_training_config_ui():
    """Buat komponen UI untuk konfigurasi training model."""
    # Container utama
    main_container = widgets.VBox([], layout=widgets.Layout(width='100%'))
    
    # Header
    header = create_header(
        "⚙️ Training Configuration",
        "Konfigurasi parameter training model SmartCash"
    )
    
    # Backbone selection section
    backbone_section = create_section_title("3.1 - Backbone Selection", "🦴")
    
    backbone_options = widgets.VBox([
        widgets.RadioButtons(
            options=['EfficientNet-B4 (Recommended)', 'CSPDarknet'],
            value='EfficientNet-B4 (Recommended)',
            description='Backbone:',
            style={'description_width': 'initial'},
        ),
        widgets.Checkbox(
            value=True,
            description='Use pretrained weights',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Freeze backbone initially',
            style={'description_width': 'initial'}
        )
    ])
    
    backbone_info = create_info_alert(
        "EfficientNet-B4 memberikan balance yang baik antara akurasi dan kecepatan untuk deteksi mata uang.",
        "info", "ℹ️"
    )
    
    # Model hyperparameters section
    hyperparams_section = create_section_title("3.2 - Model Hyperparameters", "🎛️")
    
    hyperparams_options = widgets.VBox([
        widgets.IntText(
            value=50,
            description='Epochs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.IntText(
            value=16,
            description='Batch Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.FloatText(
            value=0.01,
            description='Learning Rate:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Dropdown(
            options=['Adam', 'AdamW', 'SGD'],
            value='Adam',
            description='Optimizer:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Dropdown(
            options=['cosine', 'linear', 'step'],
            value='cosine',
            description='Scheduler:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.IntText(
            value=10,
            description='Early Stopping:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        )
    ])
    
    # Layer configuration section
    layer_section = create_section_title("3.3 - Layer Configuration", "🔍")
    
    layer_config = widgets.VBox([
        # Banknote layer
        widgets.HBox([
            widgets.Checkbox(
                value=True,
                description='Banknote Layer',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='150px')
            ),
            widgets.FloatText(
                value=0.25,
                description='Threshold:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            ),
            widgets.Label('Classes: 001, 002, 005, 010, 020, 050, 100')
        ]),
        
        # Nominal layer
        widgets.HBox([
            widgets.Checkbox(
                value=True,
                description='Nominal Layer',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='150px')
            ),
            widgets.FloatText(
                value=0.30,
                description='Threshold:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            ),
            widgets.Label('Classes: l2_001, l2_002, l2_005, l2_010, l2_020, l2_050, l2_100')
        ]),
        
        # Security layer
        widgets.HBox([
            widgets.Checkbox(
                value=True,
                description='Security Layer',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='150px')
            ),
            widgets.FloatText(
                value=0.35,
                description='Threshold:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            ),
            widgets.Label('Classes: l3_sign, l3_text, l3_thread')
        ])
    ])
    
    layer_info = create_info_alert(
        "Model menggunakan pendekatan multi-layer untuk deteksi banknote, nominal, dan fitur keamanan.",
        "info", "ℹ️"
    )
    
    # Training strategy section
    strategy_section = create_section_title("3.4 - Training Strategy", "🎯")
    
    strategy_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Use data augmentation during training',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable mixed precision training',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Save best model checkpoint',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Validate every epoch',
            style={'description_width': 'initial'}
        ),
        widgets.IntText(
            value=5,
            description='Save period (epochs):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        )
    ])
    
    # Tab untuk advanced params
    advanced_tabs = create_tab_view({
        'Augmentation': widgets.VBox([
            widgets.FloatSlider(value=0.5, min=0, max=1, description='Flip LR:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.0, min=0, max=1, description='Flip UD:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1.0, min=0, max=1, description='Mosaic:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.5, min=0, max=1, description='Scale:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.1, min=0, max=0.5, description='Translate:', style={'description_width': 'initial'}),
        ]),
        'Loss Weights': widgets.VBox([
            widgets.FloatSlider(value=0.05, min=0, max=0.2, step=0.01, description='Box loss:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.5, min=0, max=1.0, step=0.05, description='Obj loss:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.5, min=0, max=1.0, step=0.05, description='Cls loss:', style={'description_width': 'initial'}),
        ]),
        'Advanced': widgets.VBox([
            widgets.FloatText(value=0.0005, description='Weight decay:', style={'description_width': 'initial'}),
            widgets.FloatText(value=0.937, description='Momentum:', style={'description_width': 'initial'}),
            widgets.Checkbox(value=False, description='Use SWA', style={'description_width': 'initial'}),
            widgets.Checkbox(value=False, description='Use EMA', style={'description_width': 'initial'}),
        ])
    })
    
    # Save config button
    save_button = widgets.Button(
        description='Save Configuration',
        button_style='primary',
        icon='save'
    )
    
    reset_button = widgets.Button(
        description='Reset to Default',
        button_style='warning',
        icon='refresh'
    )
    
    status_output = widgets.Output()
    
    # Pasang semua komponen
    main_container.children = [
        header,
        # Backbone section
        backbone_section,
        backbone_options,
        backbone_info,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Hyperparameters section
        hyperparams_section,
        hyperparams_options,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Layer section
        layer_section,
        layer_config,
        layer_info,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Strategy section
        strategy_section,
        strategy_options,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Advanced settings
        create_section_title("Advanced Settings", "⚙️"),
        advanced_tabs,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Buttons
        widgets.HBox([save_button, reset_button]),
        status_output
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'backbone_options': backbone_options,
        'hyperparams_options': hyperparams_options,
        'layer_config': layer_config,
        'strategy_options': strategy_options,
        'advanced_tabs': advanced_tabs,
        'save_button': save_button,
        'reset_button': reset_button,
        'status_output': status_output
    }
    
    return ui_components
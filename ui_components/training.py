"""
File: smartcash/ui_components/training.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk eksekusi training model SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from smartcash.utils.ui_utils import (
    create_header, create_section_title, create_info_alert,
    create_status_indicator, create_loading_indicator
)

def create_training_execution_ui():
    """Buat komponen UI untuk eksekusi training model."""
    # Container utama
    main_container = widgets.VBox([], layout=widgets.Layout(width='100%'))
    
    # Header
    header = create_header(
        "🏋️ Training Execution",
        "Eksekusi training model SmartCash dengan konfigurasi terpilih"
    )
    
    # Training model section
    model_section = create_section_title("4.1 - Model Training", "🔄")
    
    # Training options
    training_options = widgets.VBox([
        widgets.Dropdown(
            options=['From Scratch', 'Continue From Last Checkpoint', 'Transfer Learning'],
            value='From Scratch',
            description='Mode Training:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.Checkbox(
            value=True,
            description='Use GPU if available',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable logging',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable checkpointing',
            style={'description_width': 'initial'}
        )
    ])
    
    # Tombol training
    start_training_button = widgets.Button(
        description='Start Training',
        button_style='primary',
        icon='play'
    )
    
    pause_training_button = widgets.Button(
        description='Pause Training',
        button_style='warning',
        icon='pause',
        disabled=True
    )
    
    stop_training_button = widgets.Button(
        description='Stop Training',
        button_style='danger',
        icon='stop',
        disabled=True
    )
    
    # Training progress
    training_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Training:',
        bar_style='info',
        orientation='horizontal'
    )
    
    # Epoch progress
    epoch_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Epoch:',
        bar_style='info',
        orientation='horizontal'
    )
    
    # Training status
    training_status = widgets.Output()
    
    # Performance tracking section
    metrics_section = create_section_title("4.2 - Performance Tracking", "📈")
    
    # Metrics output area
    metrics_output = widgets.Output()
    
    # Live chart area
    charts_output = widgets.Output()
    
    # Checkpoint management section
    checkpoint_section = create_section_title("4.3 - Checkpoint Management", "💾")
    
    checkpoint_info = widgets.Output()
    
    # Checkpoint options
    checkpoint_options = widgets.VBox([
        widgets.Dropdown(
            options=['Best Model (mAP)', 'Latest Model', 'Custom Epoch'],
            value='Best Model (mAP)',
            description='Load Model:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.Dropdown(
            options=[],  # Akan diisi dinamis
            description='Checkpoint:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%'),
            disabled=True
        ),
        widgets.Checkbox(
            value=False,
            description='Export model setelah training',
            style={'description_width': 'initial'}
        )
    ])
    
    # Tombol load dan export
    load_checkpoint_button = widgets.Button(
        description='Load Checkpoint',
        button_style='info',
        icon='download'
    )
    
    export_model_button = widgets.Button(
        description='Export Model',
        button_style='info',
        icon='share-square'
    )
    
    # Metrics visualization section
    visualization_section = create_section_title("4.4 - Live Metrics Visualization", "📊")
    
    # Visualization options
    visualization_tabs = widgets.Tab()
    visualization_tabs.children = [
        widgets.Output(),  # Loss curves
        widgets.Output(),  # Metrics history
        widgets.Output(),  # Learning rate
        widgets.Output()   # Confusion matrix
    ]
    visualization_tabs.set_title(0, 'Loss Curves')
    visualization_tabs.set_title(1, 'Metrics History')
    visualization_tabs.set_title(2, 'Learning Rate')
    visualization_tabs.set_title(3, 'Confusion Matrix')
    
    # Pasang semua komponen
    main_container.children = [
        header,
        # Training model section
        model_section,
        training_options,
        widgets.HBox([start_training_button, pause_training_button, stop_training_button]),
        training_progress,
        epoch_progress,
        training_status,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Performance tracking section
        metrics_section,
        metrics_output,
        charts_output,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Checkpoint management section
        checkpoint_section,
        checkpoint_info,
        checkpoint_options,
        widgets.HBox([load_checkpoint_button, export_model_button]),
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        # Metrics visualization section
        visualization_section,
        visualization_tabs
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'training_options': training_options,
        'start_training_button': start_training_button,
        'pause_training_button': pause_training_button,
        'stop_training_button': stop_training_button,
        'training_progress': training_progress,
        'epoch_progress': epoch_progress,
        'training_status': training_status,
        'metrics_output': metrics_output,
        'charts_output': charts_output,
        'checkpoint_info': checkpoint_info,
        'checkpoint_options': checkpoint_options,
        'load_checkpoint_button': load_checkpoint_button,
        'export_model_button': export_model_button,
        'visualization_tabs': visualization_tabs
    }
    
    return ui_components
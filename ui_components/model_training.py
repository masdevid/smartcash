"""
File: smartcash/ui_components/model_training.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk eksekusi training model SmartCash.
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import (
    create_component_header,
    create_section_title,
    create_info_box,
    create_info_alert,
    styled_html
)

def create_training_execution_ui():
    """
    Buat komponen UI untuk eksekusi training model.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Training Execution",
        "Eksekusi training model SmartCash dengan konfigurasi terpilih",
        "🏋️"
    )
    
    # Training options section
    training_section = create_section_title("4.1 - Model Training", "🚀")
    
    # Training options
    training_options = widgets.VBox([
        widgets.Dropdown(
            options=['From Scratch', 'Continue From Last Checkpoint', 'Transfer Learning'],
            value='From Scratch',
            description='Mode:',
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
            description='Enable TensorBoard logging',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable checkpointing',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable early stopping',
            style={'description_width': 'initial'}
        )
    ])
    
    # Buttons for training control
    control_buttons = widgets.HBox([
        widgets.Button(
            description='Start Training',
            button_style='primary',
            icon='play',
            layout=widgets.Layout(margin='0 10px 0 0')
        ),
        widgets.Button(
            description='Pause Training',
            button_style='warning',
            icon='pause',
            layout=widgets.Layout(margin='0 10px 0 0'),
            disabled=True
        ),
        widgets.Button(
            description='Stop Training',
            button_style='danger',
            icon='stop',
            layout=widgets.Layout(margin='0'),
            disabled=True
        )
    ])
    
    # Progress tracking
    training_progress = widgets.VBox([
        widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='0/0 Epochs:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%')
        ),
        widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='0/0 Batches:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%')
        )
    ])
    
    # Training status output
    training_status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='200px',
            overflow='auto',
            margin='10px 0'
        )
    )
    
    # Metrics section
    metrics_section = create_section_title("4.2 - Performance Tracking", "📈")
    
    # Metrics display
    metrics_display = widgets.Output(
        layout=widgets.Layout(
            margin='10px 0'
        )
    )
    
    # Metrics chart tabs
    metrics_tabs = widgets.Tab()
    metrics_tabs.children = [
        widgets.Output(),  # Loss chart
        widgets.Output(),  # Metrics chart
        widgets.Output(),  # Learning rate chart
        widgets.Output()   # Class metrics chart
    ]
    metrics_tabs.set_title(0, 'Loss Curves')
    metrics_tabs.set_title(1, 'Precision/Recall')
    metrics_tabs.set_title(2, 'Learning Rate')
    metrics_tabs.set_title(3, 'Per-Class Metrics')
    
    # Checkpoint management section
    checkpoint_section = create_section_title("4.3 - Checkpoint Management", "💾")
    
    # Checkpoint info
    checkpoint_info = widgets.Output(
        layout=widgets.Layout(
            margin='10px 0'
        )
    )
    
    # Checkpoint list and options
    checkpoint_options = widgets.VBox([
        widgets.Dropdown(
            options=['Best Model (mAP)', 'Latest Epoch', 'Custom Epoch'],
            value='Best Model (mAP)',
            description='Load from:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.Select(
            options=[],
            description='Checkpoint:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', height='100px'),
            disabled=True
        )
    ])
    
    # Export options
    export_options = widgets.VBox([
        widgets.Dropdown(
            options=['ONNX', 'TorchScript', 'TensorRT'],
            value='ONNX',
            description='Format:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.Text(
            value='exports/smartcash_model',
            description='Path:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Checkpoint buttons
    checkpoint_buttons = widgets.HBox([
        widgets.Button(
            description='Load Model',
            button_style='info',
            icon='upload',
            layout=widgets.Layout(margin='0 10px 0 0')
        ),
        widgets.Button(
            description='Export Model',
            button_style='success',
            icon='download',
            layout=widgets.Layout(margin='0')
        )
    ])
    
    # Info box with helpful information
    info_box = create_info_box(
        "Training Information",
        """
        <p>Training proses menggunakan parameter yang dikonfigurasi di step sebelumnya:</p>
        <ul>
            <li><strong>Backbone:</strong> EfficientNet-B4 atau CSPDarknet (sesuai pilihan)</li>
            <li><strong>Multi-layer Detection:</strong> Banknote, Nominal, Security</li>
            <li><strong>Batch Size:</strong> Disesuaikan dengan ketersediaan memory (auto-scaling)</li>
            <li><strong>Early Stopping:</strong> Training akan berhenti otomatis jika tidak ada improvement</li>
        </ul>
        <p>Checkpoint akan disimpan secara otomatis selama training.</p>
        """,
        'info'
    )
    
    # Assemble all components
    main_container.children = [
        header,
        training_section,
        training_options,
        control_buttons,
        training_progress,
        training_status,
        metrics_section,
        metrics_display,
        metrics_tabs,
        checkpoint_section,
        checkpoint_info,
        checkpoint_options,
        export_options,
        checkpoint_buttons,
        info_box
    ]
    
    # Dictionary for component access
    ui_components = {
        'ui': main_container,
        'training_options': training_options,
        'start_button': control_buttons.children[0],
        'pause_button': control_buttons.children[1],
        'stop_button': control_buttons.children[2],
        'epoch_progress': training_progress.children[0],
        'batch_progress': training_progress.children[1],
        'training_status': training_status,
        'metrics_display': metrics_display,
        'metrics_tabs': metrics_tabs,
        'checkpoint_info': checkpoint_info,
        'checkpoint_selector': checkpoint_options.children[1],
        'load_model_button': checkpoint_buttons.children[0],
        'export_model_button': checkpoint_buttons.children[1],
        'export_format': export_options.children[0],
        'export_path': export_options.children[1]
    }
    
    return ui_components
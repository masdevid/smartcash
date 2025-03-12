"""
File: smartcash/ui_components/training_strategy.py
Author: Refactor
Deskripsi: Komponen UI untuk konfigurasi strategi training model SmartCash.
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import (
    create_component_header, 
    create_section_title,
    create_info_box
)

def create_training_strategy_ui():
    """Buat komponen UI untuk konfigurasi strategi training model."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Training Strategy",
        "Konfigurasi strategi dan teknik optimasi untuk training model SmartCash",
        "🎯"
    )
    
    # Augmentation strategy section
    augmentation_section = create_section_title("Data Augmentation Strategy", "🔄")
    
    augmentation_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Enable data augmentation',
            style={'description_width': 'initial'}
        ),
        widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.05,
            description='Mosaic prob:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.05,
            description='Flip prob:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatSlider(
            value=0.3,
            min=0,
            max=1.0,
            step=0.05,
            description='Scale jitter:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=False,
            description='Enable mixup',
            style={'description_width': 'initial'}
        )
    ])
    
    # Optimization strategy section
    optimization_section = create_section_title("Optimization Strategy", "⚙️")
    
    optimization_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Use mixed precision (FP16)',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Use Exponential Moving Average (EMA)',
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='10px 0')
        ),
        widgets.Checkbox(
            value=False,
            description='Use Stochastic Weight Averaging (SWA)',
            style={'description_width': 'initial'}
        ),
        widgets.Dropdown(
            options=['cosine', 'linear', 'step', 'constant'],
            value='cosine',
            description='LR schedule:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%', margin='10px 0')
        ),
        widgets.FloatText(
            value=0.01,
            description='Weight decay:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    ])
    
    # Training policy section
    policy_section = create_section_title("Training Policy", "📋")
    
    policy_options = widgets.VBox([
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
        ),
        widgets.IntSlider(
            value=15,
            min=5,
            max=30,
            description='Early stopping patience:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=True,
            description='Validate every epoch',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Log to TensorBoard',
            style={'description_width': 'initial'}
        )
    ])
    
    # Strategy summary
    strategy_summary = widgets.Output(
        layout=widgets.Layout(
            margin='20px 0',
            border='1px solid #ddd',
            padding='10px'
        )
    )
    
    # Buttons container
    buttons_container = widgets.HBox([
        widgets.Button(
            description='Save Strategy',
            button_style='primary',
            icon='save',
            layout=widgets.Layout(margin='0 10px 0 0')
        ),
        widgets.Button(
            description='Reset to Default',
            button_style='warning',
            icon='refresh'
        )
    ], layout=widgets.Layout(margin='15px 0'))
    
    # Status output
    status_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Info box with additional details
    info_box = create_info_box(
        "About Training Strategies",
        """
        <p>Strategi training yang tepat dapat meningkatkan performa model secara signifikan:</p>
        <ul>
            <li><strong>Data Augmentation</strong> - Memperbanyak variasi data training dengan transformasi</li>
            <li><strong>Optimization</strong> - Teknik untuk mempercepat training dan mencapai konvergensi lebih baik</li>
            <li><strong>Training Policy</strong> - Pengaturan untuk proses dan monitoring training</li>
        </ul>
        <p><strong>Rekomendasi:</strong> Gunakan kombinasi mosaic augmentation (0.5-0.8), EMA, dan cosine scheduler untuk kasus SmartCash.</p>
        """,
        'info',
        collapsed=True
    )
    
    # Assemble all components
    main_container.children = [
        header,
        augmentation_section,
        augmentation_options,
        optimization_section,
        optimization_options,
        policy_section,
        policy_options,
        strategy_summary,
        buttons_container,
        status_output,
        info_box
    ]
    
    # Create dictionary of components for handlers
    ui_components = {
        'ui': main_container,
        'augmentation_options': augmentation_options,
        'optimization_options': optimization_options,
        'policy_options': policy_options,
        'strategy_summary': strategy_summary,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'status_output': status_output
    }
    
    return ui_components
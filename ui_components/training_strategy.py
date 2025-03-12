"""
File: smartcash/ui_components/training_strategy.py
Author: Refactored
Deskripsi: Komponen UI untuk konfigurasi strategi training model SmartCash.
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import (
    create_component_header, 
    create_section_title,
    create_info_alert,
    create_info_box
)

def create_training_strategy_ui():
    """
    Buat komponen UI untuk konfigurasi strategi training model.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Training Strategy",
        "Konfigurasi strategi dan teknik optimasi untuk training model SmartCash",
        "🎯"
    )
    
    # Augmentation strategy section
    augmentation_section = create_section_title("3.3.1 - Data Augmentation Strategy", "🔄")
    
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
    
    augmentation_info = create_info_alert(
        "Augmentasi on-the-fly selama training untuk meningkatkan variasi data dan mencegah overfitting.",
        "info", "ℹ️"
    )
    
    # Optimization strategy section
    optimization_section = create_section_title("3.3.2 - Optimization Strategy", "⚙️")
    
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
    policy_section = create_section_title("3.3.3 - Training Policy", "📋")
    
    policy_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Save best model by mAP',
            style={'description_width': 'initial'}
        ),
        widgets.IntText(
            value=5,
            description='Save checkpoint every:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.IntText(
            value=15,
            description='Early stopping patience:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        ),
        widgets.Checkbox(
            value=True,
            description='Validate every epoch',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Log metrics to TensorBoard',
            style={'description_width': 'initial'}
        )
    ])
    
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
            icon='refresh',
            layout=widgets.Layout(margin='0')
        )
    ])
    
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
    
    # Visualization area for comparison
    visualization_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='200px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Info box with additional details
    info_box = create_info_box(
        "Training Strategy Details",
        """
        <p><strong>Data Augmentation</strong></p>
        <ul>
            <li><strong>Mosaic</strong>: Menggabungkan 4 gambar training menjadi 1 untuk variasi konteks</li>
            <li><strong>Flip</strong>: Refleksi horizontal untuk variasi orientasi</li>
            <li><strong>Scale jitter</strong>: Variasi skala untuk mencegah model sensitif terhadap ukuran</li>
            <li><strong>Mixup</strong>: Mencampur 2 gambar dengan alpha blending untuk variasi lebih kompleks</li>
        </ul>
        
        <p><strong>Optimization Techniques</strong></p>
        <ul>
            <li><strong>Mixed Precision</strong>: Training dengan FP16 untuk mempercepat training & mengurangi memory usage</li>
            <li><strong>EMA</strong>: Menyimpan rata-rata parameter model untuk mengurangi noise & meningkatkan stabilitas</li>
            <li><strong>SWA</strong>: Mencari optimal weight space dengan rat-rata beberapa checkpoint</li>
            <li><strong>LR Schedule</strong>: Cosine decay mengurangi learning rate secara gradual & optimal</li>
        </ul>
        
        <p><strong>Training Policy</strong></p>
        <ul>
            <li><strong>Early stopping</strong>: Berhenti training jika tidak ada improvement setelah n epochs</li>
            <li><strong>Checkpoint</strong>: Menyimpan model & bobot pada interval tertentu</li>
            <li><strong>Validation</strong>: Evaluasi model pada dataset validasi secara berkala</li>
        </ul>
        """,
        'info',
        collapsed=True
    )
    
    # Pasang semua komponen
    main_container.children = [
        header,
        augmentation_section,
        augmentation_options,
        augmentation_info,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        optimization_section,
        optimization_options,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        policy_section,
        policy_options,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        buttons_container,
        status_output,
        visualization_output,
        info_box
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'augmentation_options': augmentation_options,
        'optimization_options': optimization_options,
        'policy_options': policy_options,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'status_output': status_output,
        'visualization_output': visualization_output
    }
    
    return ui_components
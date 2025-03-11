"""
File: smartcash/ui_components/split_config.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk konfigurasi split dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path
from smartcash.utils.ui_utils import create_component_header, create_info_box

def create_split_config_ui():
    """Buat komponen UI untuk konfigurasi split dataset."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Dataset Split Configuration",
        "Konfigurasi pembagian dataset untuk training, validation, dan testing",
        "✂️"
    )
    
    # Split options
    split_options = widgets.VBox([
        widgets.BoundedFloatText(
            value=70.0,
            min=50.0,
            max=90.0,
            description='Train %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.BoundedFloatText(
            value=15.0,
            min=5.0,
            max=30.0,
            description='Validation %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.BoundedFloatText(
            value=15.0,
            min=5.0,
            max=30.0, 
            description='Test %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=True,
            description='Stratified split (preserve class distribution)',
            style={'description_width': 'initial'}
        )
    ])
    
    # Advanced options
    advanced_options = widgets.VBox([
        widgets.IntText(
            value=42,
            description='Random seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=False,
            description='Force resplit dataset',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Preserve folder structure',
            style={'description_width': 'initial'}
        )
    ])
    
    # Advanced settings accordion
    advanced_settings = widgets.Accordion(children=[advanced_options], selected_index=None)
    advanced_settings.set_title(0, "⚙️ Advanced Split Settings")
    
    # Split button and status
    split_button = widgets.Button(
        description='Apply Split',
        button_style='primary',
        icon='scissors'
    )
    
    split_status = widgets.Output()
    
    # Dataset stats output
    stats_output = widgets.Output()
    
    # Info box
    info_box = create_info_box(
        "Tentang Split Dataset",
        """
        <p>Pembagian dataset menjadi 3 subset:</p>
        <ul>
            <li><strong>Train</strong>: Dataset untuk training model</li>
            <li><strong>Validation</strong>: Dataset untuk validasi selama training</li>
            <li><strong>Test</strong>: Dataset untuk evaluasi final model</li>
        </ul>
        <p>Gunakan <strong>stratified split</strong> untuk memastikan distribusi kelas tetap seimbang di semua subset.</p>
        """,
        'info'
    )
    
    # Pasang semua komponen
    main_container.children = [
        header,
        info_box,
        split_options,
        advanced_settings,
        widgets.HBox([split_button]),
        split_status,
        stats_output
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'split_options': split_options,
        'advanced_options': advanced_options,
        'split_button': split_button,
        'split_status': split_status,
        'stats_output': stats_output
    }
    
    return ui_components
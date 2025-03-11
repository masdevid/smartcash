"""
File: smartcash/ui_components/preprocessing.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk preprocessing dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path
from smartcash.utils.ui_utils import create_component_header, create_info_box

def create_preprocessing_ui():
    """Buat komponen UI untuk preprocessing dataset."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Dataset Preprocessing",
        "Preprocessing dataset untuk training model SmartCash",
        "🔧"
    )
    
    # Preprocessing options
    preprocess_options = widgets.VBox([
        widgets.IntRangeSlider(
            value=[640, 640],
            min=320,
            max=1280,
            step=32,
            description='Image size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=True,
            description='Enable normalization',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Enable caching',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=4,
            min=1,
            max=16,
            description='Workers:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    ])
    
    # Preprocess button and status
    preprocess_button = widgets.Button(
        description='Run Preprocessing',
        button_style='primary',
        icon='cog'
    )
    
    # Create output for logs
    preprocess_status = widgets.Output(
        layout=widgets.Layout(
            max_height='300px', 
            overflow='auto'
        )
    )
    
    # Collapsible log output
    log_accordion = widgets.Accordion(
        children=[preprocess_status],
        selected_index=None
    )
    log_accordion.set_title(0, "📋 Preprocessing Logs")
    
    preprocess_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout={'visibility': 'hidden'}
    )
    
    # Info box
    info_box = create_info_box(
        "Tentang Preprocessing",
        """
        <p>Preprocessing meliputi:</p>
        <ul>
            <li><strong>Resize</strong>: Ubah ukuran gambar menjadi ukuran yang seragam</li>
            <li><strong>Normalization</strong>: Normalisasi pixel values untuk training yang lebih stabil</li>
            <li><strong>Caching</strong>: Simpan gambar yang sudah diproses untuk mempercepat loading</li>
        </ul>
        """,
        'info'
    )
    
    # Pasang semua komponen
    main_container.children = [
        header,
        info_box,
        preprocess_options,
        widgets.HBox([preprocess_button]),
        preprocess_progress,
        log_accordion
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'preprocess_options': preprocess_options,
        'preprocess_button': preprocess_button,
        'preprocess_progress': preprocess_progress,
        'preprocess_status': preprocess_status,
        'log_accordion': log_accordion
    }
    
    return ui_components
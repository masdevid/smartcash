"""
File: smartcash/ui_components/preprocessing.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk preprocessing dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

def create_preprocessing_ui():
    """Buat komponen UI untuk preprocessing dataset."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("<h2>🔧 Dataset Preprocessing</h2><p>Preprocessing dataset untuk training model SmartCash</p>")
    
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
    
    # Cache settings
    cache_settings = widgets.VBox([
        widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Cache size (GB):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%')
        ),
        widgets.IntSlider(
            value=24,
            min=1,
            max=168,
            description='TTL (hours):',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%')
        ),
        widgets.Checkbox(
            value=True,
            description='Auto cleanup',
            style={'description_width': 'initial'}
        )
    ])
    
    # Preprocess button and status
    preprocess_button = widgets.Button(
        description='Run Preprocessing',
        button_style='primary',
        icon='cog'
    )
    
    preprocess_status = widgets.Output()
    preprocess_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout={'visibility': 'hidden'}
    )
    
    # Advanced settings accordion
    advanced_settings = widgets.Accordion(children=[cache_settings], selected_index=None)
    advanced_settings.set_title(0, "🔧 Advanced Cache Settings")
    
    # Info box
    info_box = widgets.HTML("""
    <div style="padding: 10px; background-color: #d1ecf1; border-left: 4px solid #0c5460; 
             color: #0c5460; margin: 10px 0; border-radius: 4px;">
        <h4 style="margin-top: 0;">ℹ️ Tentang Preprocessing</h4>
        <p>Preprocessing meliputi:</p>
        <ul>
            <li><strong>Resize</strong>: Ubah ukuran gambar menjadi ukuran yang seragam</li>
            <li><strong>Normalization</strong>: Normalisasi pixel values untuk training yang lebih stabil</li>
            <li><strong>Caching</strong>: Simpan gambar yang sudah diproses untuk mempercepat loading</li>
        </ul>
    </div>
    """)
    
    # Pasang semua komponen
    main_container.children = [
        header,
        info_box,
        preprocess_options,
        advanced_settings,
        widgets.HBox([preprocess_button]),
        preprocess_progress,
        preprocess_status
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'preprocess_options': preprocess_options,
        'cache_settings': cache_settings,
        'preprocess_button': preprocess_button,
        'preprocess_progress': preprocess_progress,
        'preprocess_status': preprocess_status
    }
    
    return ui_components
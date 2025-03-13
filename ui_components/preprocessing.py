"""
File: smartcash/ui_components/preprocessing.py
Author: Alfrida Sabar (revisi)
Deskripsi: Komponen UI untuk preprocessing dataset SmartCash dengan cleanup button dan config persistence.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

from utils.ui_utils import create_info_box

def create_preprocessing_ui():
    """
    Buat komponen UI untuk preprocessing dataset.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("""
    <div style="background-color: #f0f8ff; padding: 15px; color: black; 
              border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #3498db;">
        <h2 style="color: inherit; margin-top: 0;">🔧 Dataset Preprocessing</h2>
        <p style="color: inherit; margin-bottom: 0;">Preprocessing dataset untuk training model SmartCash</p>
    </div>
    """)
    
    # Preprocessing options
    preprocess_options = widgets.VBox([
        widgets.IntRangeSlider(
            value=[640, 640],
            min=320,
            max=640,
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
    
    # Advanced options accordion
    validation_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Validate dataset integrity',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Fix issues automatically',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Move invalid files',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='data/invalid',
            description='Invalid dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    ])
    
    advanced_accordion = widgets.Accordion(children=[validation_options])
    advanced_accordion.set_title(0, "🔍 Validation Options")
    
    # Split selection for preprocessing
    split_selector = widgets.RadioButtons(
        options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
        value='All Splits',
        description='Process:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Preprocess button and status
    preprocess_button = widgets.Button(
        description='Run Preprocessing',
        button_style='primary',
        icon='cog',
        layout=widgets.Layout(width='auto')
    )
    
    stop_button = widgets.Button(
        description='Stop',
        button_style='danger',
        icon='stop',
        layout=widgets.Layout(width='auto', display='none')
    )
    
    cleanup_button = widgets.Button(
        description='Clean Preprocessed Data',
        button_style='danger',
        icon='trash',
        layout=widgets.Layout(display='none')
    )
    
    button_container = widgets.HBox([preprocess_button, stop_button], 
                                   layout=widgets.Layout(margin='10px 0'))
    
    # Progress tracking
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Overall:',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    current_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Current:',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    progress_container = widgets.VBox([
        widgets.HTML("<h4>📊 Progress</h4>"),
        progress_bar,
        current_progress
    ])
    
    # Create output for logs
    preprocess_status = widgets.Output(
        layout=widgets.Layout(
            max_height='300px', 
            overflow='auto',
            border='1px solid #ddd',
            margin='10px 0'
        )
    )
    
    # Collapsible log output
    log_accordion = widgets.Accordion(
        children=[preprocess_status],
        selected_index=None,
        layout=widgets.Layout(margin='10px 0')
    )
    log_accordion.set_title(0, "📋 Preprocessing Logs")
    
    # Summary stats (akan ditampilkan setelah preprocessing)
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd', 
            padding='10px', 
            margin='10px 0', 
            display='none'
        )
    )
    
    # Info box
    info_box = create_info_box("""
    <div style="padding: 10px; background-color: #d1ecf1; border-left: 4px solid #0c5460; 
             color: #0c5460; margin: 10px 0; border-radius: 4px;">
        <h4 style="margin-top: 0; color: inherit;">ℹ️ Tentang Preprocessing</h4>
        <p>Preprocessing meliputi:</p>
        <ul>
            <li><strong>Resize</strong>: Ubah ukuran gambar menjadi ukuran yang seragam</li>
            <li><strong>Normalization</strong>: Normalisasi pixel values untuk training yang lebih stabil</li>
            <li><strong>Caching</strong>: Simpan gambar yang sudah diproses untuk mempercepat loading</li>
            <li><strong>Validation</strong>: Validasi integritas dataset, cek label dan gambar rusak</li>
        </ul>
        <p><strong>📝 Konfigurasi</strong> akan otomatis disimpan ke <code>configs/preprocessing_config.yaml</code></p>
    </div>
    """, 'info', collapsed=True)
    
    # Cleanup container
    cleanup_container = widgets.HBox([cleanup_button], layout=widgets.Layout(margin='10px 0'))
    
    # Pasang semua komponen
    main_container.children = [
        header,
        info_box,
        widgets.HTML("<h4>⚙️ Preprocessing Settings</h4>"),
        preprocess_options,
        split_selector,
        advanced_accordion,
        button_container,
        progress_container,
        log_accordion,
        summary_container,
        cleanup_container
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'preprocess_options': preprocess_options,
        'validation_options': validation_options,
        'split_selector': split_selector,
        'preprocess_button': preprocess_button,
        'stop_button': stop_button,
        'cleanup_button': cleanup_button,
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'preprocess_status': preprocess_status,
        'log_accordion': log_accordion,
        'summary_container': summary_container
    }
    
    return ui_components
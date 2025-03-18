"""
File: smartcash/ui/dataset/preprocessing_component.py
Deskripsi: Komponen UI untuk preprocessing dataset SmartCash dengan accordion terbuka dan opsi yang sesuai
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_preprocessing_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk preprocessing dataset dengan design pattern yang konsisten.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI
    from smartcash.ui.components.headers import create_header
    from smartcash.ui.components.alerts import create_info_box
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Header
    header = create_header(
        f"{ICONS['processing']} Dataset Preprocessing",
        "Preprocessing dataset untuk training model SmartCash"
    )
    
    # Panel info status
    status_panel = widgets.HTML(value=f"""
        <div style="padding: 10px; background-color: {COLORS['alert_info_bg']}; 
                    color: {COLORS['alert_info_text']}; margin: 10px 0; border-radius: 4px; 
                    border-left: 4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Konfigurasi preprocessing dataset</p>
        </div>
    """)
    
    # Preprocessing options (sesuaikan dengan parameter dataset manager)
    preprocess_options = widgets.VBox([
        widgets.IntSlider(
            value=640,
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
            description='Preserve aspect ratio',
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
    
    # Advanced options - Validation options
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
            layout=widgets.Layout(width='60%')
        )
    ])
    
    advanced_accordion = widgets.Accordion(children=[validation_options])
    advanced_accordion.set_title(0, f"{ICONS['search']} Validation Options")
    
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
        layout=widgets.Layout(width='auto', display='none')
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
        widgets.HTML(f"<h4>{ICONS['stats']} Progress</h4>"),
        progress_bar,
        current_progress
    ])
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Collapsible log output - Sekarang default terbuka (selected_index=0)
    log_accordion = widgets.Accordion(
        children=[status],
        selected_index=0,  # Set terbuka secara default
        layout=widgets.Layout(margin='10px 0')
    )
    log_accordion.set_title(0, f"{ICONS['file']} Preprocessing Logs")
    
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
    help_panel = create_info_box(
        f"{ICONS['info']} Tentang Preprocessing",
        """
        <p>Preprocessing meliputi beberapa langkah penting:</p>
        <ul>
            <li><strong>Resize</strong>: Ubah ukuran gambar menjadi ukuran yang seragam</li>
            <li><strong>Normalization</strong>: Normalisasi pixel values untuk training yang lebih stabil</li>
            <li><strong>Caching</strong>: Simpan gambar yang sudah diproses untuk mempercepat loading</li>
            <li><strong>Validation</strong>: Validasi integritas dataset, cek label dan gambar rusak</li>
        </ul>
        <p><strong>📝 Konfigurasi</strong> akan otomatis disimpan ke <code>configs/preprocessing_config.yaml</code></p>
        
        <p><strong>Catatan:</strong> Gambar yang telah dipreprocessing akan disimpan di direktori 
        <code>data/preprocessed/[split]</code> untuk penggunaan berikutnya.</p>
        """,
        'info',
        collapsed=True
    )
    
    # Cleanup container
    cleanup_container = widgets.HBox([cleanup_button], layout=widgets.Layout(margin='10px 0'))
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4>{ICONS['settings']} Preprocessing Settings</h4>"),
        preprocess_options,
        split_selector,
        advanced_accordion,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        button_container,
        progress_container,
        log_accordion,
        summary_container,
        cleanup_container,
        help_panel
    ])
    
    # Komponen UI
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'preprocess_options': preprocess_options,
        'validation_options': validation_options,
        'split_selector': split_selector,
        'preprocess_button': preprocess_button,
        'stop_button': stop_button,
        'cleanup_button': cleanup_button,
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'module_name': 'preprocessing'
    }
    
    return ui_components
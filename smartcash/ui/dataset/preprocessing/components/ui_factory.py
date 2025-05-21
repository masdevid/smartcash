"""
File: smartcash/ui/dataset/preprocessing/components/ui_factory.py
Deskripsi: Factory untuk komponen UI preprocessing dataset dengan perbaikan duplikasi opsi dan tampilan tombol
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk preprocessing dataset.
    
    Args:
        config: Konfigurasi preprocessing (opsional)
        
    Returns:
        Dictionary berisi semua komponen UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import create_divider
    from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info
    
    # Import shared components
    from smartcash.ui.components.split_selector import create_split_selector
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.validation_options import create_validation_options
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    from smartcash.ui.components.sync_info_message import create_sync_info_message
    
    # Config default jika tidak ada
    if not config:
        config = {}
    
    # Header preprocessing
    header = create_header(
        f"{ICONS['processing']} Dataset Preprocessing", 
        "Preprocessing dataset untuk training model SmartCash"
    )
    
    # Panel status
    status_panel = create_status_panel(
        "Konfigurasi preprocessing dataset", "info"
    )
    
    # Create preprocessing options
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_options
    preprocess_options = create_preprocessing_options(config)
    
    # Opsi validasi preprocessing
    validation_options = create_validation_options(config)
    
    # Tombol aksi
    action_buttons = create_action_buttons(
        primary_label="Preprocess Dataset",
        primary_icon="play",
        secondary_buttons=[
            ("Berhenti", "stop", "warning")
        ],
        cleanup_enabled=True
    )
    
    # Default sembunyikan tombol stop
    if 'stop_button' in action_buttons and hasattr(action_buttons['stop_button'], 'layout'):
        action_buttons['stop_button'].layout.display = 'none'
    
    # Tombol save & reset
    save_reset_buttons = create_save_reset_buttons()
    
    # Info sinkronisasi
    sync_info = create_sync_info_message()
    
    # Worker dan target split
    worker_slider = widgets.IntSlider(
        value=config.get('num_workers', 4),
        min=1,
        max=8,
        step=1,
        description='Workers:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    split_selector = create_split_selector(
        selected_value=config.get('split', 'All Splits'),
        description="Split:",
        width='95%'
    )
    
    # UI untuk menampilkan advanced options di kolom kedua
    advanced_options_column = widgets.VBox([
        widgets.HTML(f"<h5 style='margin-bottom: 10px; color: {COLORS['dark']};'>{ICONS['config']} Opsi Preprocessing</h5>"),
        widgets.HTML(f"<div style='margin-bottom: 8px;'><b>Worker Thread:</b> Jumlah thread untuk preprocessing</div>"),
        worker_slider,
        widgets.HTML(f"<div style='margin: 8px 0;'><b>Target Split:</b> Bagian dataset yang akan diproses</div>"),
        split_selector
    ], layout=widgets.Layout(padding='10px 5px', width='48%'))
    
    # Area konfirmasi dengan styling yang lebih baik - hidden by default
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            width='100%', 
            margin='15px 0',
            min_height='30px',
            border='1px solid #ddd',
            padding='10px',
            display='none',  # Hidden by default
            border_radius='5px'
        )
    )
    
    # Progress tracking
    progress_components = create_progress_tracking(
        module_name='preprocessing',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )
    
    # Log accordion
    log_components = create_log_accordion(
        module_name='preprocessing',
        height='200px',
        width='100%'
    )
    
    # Help panel
    help_panel = get_preprocessing_info()
    
    # Layout 2 kolom untuk input options
    options_container = widgets.HBox([
        preprocess_options,
        advanced_options_column
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Rakit komponen UI utama
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Preprocessing Settings</h4>"),
        options_container,  # 2 kolom untuk options
        widgets.VBox([
            save_reset_buttons['container'],
            sync_info['container']
        ], layout=widgets.Layout(align_items='flex-end', width='100%')),
        create_divider(),
        action_buttons['container'],
        confirmation_area,  # Area konfirmasi di bawah tombol aksi (tersembunyi)
        progress_components['progress_container'],
        log_components['log_accordion'],
        help_panel
    ])
    
    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'preprocess_options': preprocess_options,
        'validation_options': validation_options,
        'split_selector': split_selector,  # Gunakan split selector dari kolom kedua
        'worker_slider': worker_slider,  # Referensi langsung ke worker slider
        'preprocess_button': action_buttons['primary_button'],
        'stop_button': action_buttons['stop_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_buttons': save_reset_buttons,
        'sync_info': sync_info,
        'action_buttons': action_buttons,  # Menyimpan semua action buttons
        'button_container': action_buttons['container'],
        'confirmation_area': confirmation_area,
        'module_name': 'preprocessing',
        'data_dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    }
    
    # Tambahkan komponen progress tracking
    ui_components.update({
        'progress_bar': progress_components['progress_bar'],
        'progress_container': progress_components['progress_container'],
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label')
    })
    
    # Tambahkan komponen log
    ui_components.update({
        'status': log_components['log_output'],
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion']
    })
    
    return ui_components
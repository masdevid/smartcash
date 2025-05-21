"""
File: smartcash/ui/dataset/preprocessing/components/ui_factory.py
Deskripsi: Factory untuk komponen UI preprocessing dataset
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
    
    # Import komponen preprocessing spesifik
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_options
    
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
    
    # Opsi preprocessing (resolusi, normalisasi, dll)
    preprocess_options = create_preprocessing_options(config)
    
    # Opsi validasi
    validation_options = create_validation_options(
        title="Opsi Validasi",
        description="Pilih opsi validasi yang akan dijalankan selama preprocessing",
        options=[
            ("Validasi format gambar", "validate_image_format", True),
            ("Validasi label format", "validate_label_format", True),
            ("Validasi dimensi gambar", "validate_image_dimensions", True),
            ("Validasi bounding box", "validate_bounding_box", True)
        ],
        width="100%",
        icon="validation"
    )
    
    # Accordion untuk opsi validasi (selalu tertutup di awal)
    advanced_accordion = widgets.Accordion(
        children=[validation_options['container']], 
        selected_index=None
    )
    advanced_accordion.set_title(0, f"{ICONS['search']} Validation Options")
    
    # Tombol save dan reset
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi preprocessing dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi preprocessing ke default",
        save_icon="save",
        reset_icon="reset",
        with_sync_info=False,
        button_width="100px",
        container_width="100%"
    )
    
    # Pesan sinkronisasi
    sync_info = create_sync_info_message(
        message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan.",
        icon="info",
        color="#666",
        font_style="italic",
        margin_top="5px",
        width="100%"
    )
    
    # Tombol aksi (run, stop, cleanup)
    action_buttons = create_action_buttons(
        primary_label="Run Preprocessing",
        primary_icon="cog",
        cleanup_enabled=True
    )
    
    # Area konfirmasi
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            width='100%', 
            margin='10px 0',
            border='1px solid #ddd',
            padding='10px',
            display='none'  # Hidden by default
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
    
    # Rakit komponen UI utama
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Preprocessing Settings</h4>"),
        preprocess_options,
        advanced_accordion,
        widgets.VBox([
            save_reset_buttons['container'],
            sync_info['container']
        ], layout=widgets.Layout(align_items='flex-end', width='100%')),
        create_divider(),
        action_buttons['container'],
        confirmation_area,  # Area konfirmasi di bawah tombol
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
        'split_selector': preprocess_options.target_split,
        'advanced_accordion': advanced_accordion,
        'preprocess_button': action_buttons['primary_button'],
        'stop_button': action_buttons['stop_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_buttons': save_reset_buttons,
        'sync_info': sync_info,
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
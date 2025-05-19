"""
File: smartcash/ui/dataset/preprocessing/components/preprocessing_component.py
Deskripsi: Komponen UI utama untuk preprocessing dataset menggunakan shared components
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_preprocessing_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk preprocessing dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info
    from smartcash.ui.utils.layout_utils import create_divider
    
    # Import shared components
    from smartcash.ui.components.split_selector import create_split_selector
    from smartcash.ui.components.action_buttons import create_action_buttons, create_visualization_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.model_info_panel import create_model_info_panel
    from smartcash.ui.components.feature_checkbox_group import create_feature_checkbox_group
    from smartcash.ui.components.config_form import create_config_form
    
    # Import komponen submodules preprocessing
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_options
    from smartcash.ui.dataset.preprocessing.components.validation_options import create_validation_options
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['processing']} Dataset Preprocessing", 
                          "Preprocessing dataset untuk training model SmartCash")
    
    # Panel info status
    status_panel = create_status_panel("Konfigurasi preprocessing dataset", "info")
    
    # Preprocessing options (split dari komponen besar)
    preprocess_options = create_preprocessing_options(config)
    
    # Split selector menggunakan shared component
    split_selector = create_split_selector(
        selected_value='Train Only',
        description="Target Split:",
        width='100%',
        icon='split'
    )
    
    # Validation options menggunakan shared component feature_checkbox_group
    validation_features = [
        ("Validasi format gambar", True),
        ("Validasi label format", True),
        ("Validasi dimensi gambar", True),
        ("Validasi bounding box", True)
    ]
    
    validation_options_group = create_feature_checkbox_group(
        features=validation_features,
        title="Opsi Validasi",
        description="Pilih opsi validasi yang akan dijalankan selama preprocessing",
        width="100%",
        icon="search"
    )
    
    # Gunakan komponen lama untuk kompatibilitas
    validation_options = create_validation_options(config)
    
    # Accordion untuk validation options - selalu tertutup di awal
    advanced_accordion = widgets.Accordion(children=[validation_options_group['container']], selected_index=None)
    advanced_accordion.set_title(0, f"{ICONS['search']} Validation Options")
    
    # Buat tombol-tombol preprocessing dengan shared component
    action_buttons = create_action_buttons(
        primary_label="Run Preprocessing",
        primary_icon="cog",
        cleanup_enabled=True
    )
    
    # Buat tombol visualisasi dengan shared component
    visualization_buttons = create_visualization_buttons()
    
    # Progress tracking dengan shared component
    progress_components = create_progress_tracking(
        module_name='preprocessing',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )
    
    # Log accordion dengan shared component
    log_components = create_log_accordion(
        module_name='preprocessing',
        height='200px',
        width='100%'
    )
    
    # Summary stats container dengan styling yang konsisten
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd', 
            padding='10px', 
            margin='10px 0', 
            display='none'
        )
    )
    
    # Container visualisasi
    visualization_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd', 
            padding='10px', 
            margin='10px 0', 
            display='none'
        )
    )
    
    # Help panel dengan komponen info_box standar
    help_panel = get_preprocessing_info()
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Preprocessing Settings</h4>"),
        preprocess_options,
        split_selector,
        advanced_accordion,
        create_divider(),
        action_buttons['container'],
        progress_components['progress_container'],
        log_components['log_accordion'],
        summary_container,
        help_panel
    ])
    
    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'preprocess_options': preprocess_options,
        'validation_options': validation_options,
        'validation_options_group': validation_options_group,
        'split_selector': split_selector,
        'advanced_accordion': advanced_accordion,
        'preprocess_button': action_buttons['primary_button'],
        'preprocessing_button': action_buttons['primary_button'],  # Alias untuk kompatibilitas
        'stop_button': action_buttons['stop_button'],
        'reset_button': action_buttons['reset_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': action_buttons['save_button'],
        'button_container': action_buttons['container'],
        'visualization_buttons': visualization_buttons['container'],
        'visualize_button': visualization_buttons['visualize_button'],
        'compare_button': visualization_buttons['compare_button'],
        'distribution_button': visualization_buttons.get('distribution_button'),
        'summary_container': summary_container,
        'visualization_container': visualization_container,
        'module_name': 'preprocessing',
        # Default dataset paths
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
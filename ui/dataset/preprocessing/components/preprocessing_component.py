"""
File: smartcash/ui/dataset/preprocessing/components/preprocessing_component.py
Deskripsi: Komponen UI utama untuk preprocessing dataset
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
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET
    from smartcash.ui.handlers.status_handler import create_status_panel
    
    # Import komponen submodules preprocessing
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_options
    from smartcash.ui.dataset.preprocessing.components.validation_options import create_validation_options
    from smartcash.ui.dataset.preprocessing.components.split_selector import create_split_selector
    
    # Import komponen button standar
    from smartcash.ui.components.action_buttons import create_action_buttons, create_visualization_buttons
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['processing']} Dataset Preprocessing", 
                          "Preprocessing dataset untuk training model SmartCash")
    
    # Panel info status
    status_panel = create_status_panel("Konfigurasi preprocessing dataset", "info")
    
    # Preprocessing options (split dari komponen besar)
    preprocess_options = create_preprocessing_options(config)
    split_selector = create_split_selector()
    
    # Validation options dalam accordion
    validation_options = create_validation_options(config)
    
    # Accordion untuk validation options - selalu tertutup di awal
    advanced_accordion = widgets.Accordion(children=[validation_options], selected_index=None)
    advanced_accordion.set_title(0, f"{ICONS['search']} Validation Options")
    
    # Buat tombol-tombol preprocessing dengan komponen standar
    action_buttons = create_action_buttons(
        primary_label="Run Preprocessing",
        primary_icon="cog",
        cleanup_enabled=True
    )
    
    # Buat tombol visualisasi meskipun tidak ditampilkan di UI utama
    visualization_buttons = create_visualization_buttons()
    
    # Progress tracking dengan styling standar
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Total:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Split:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    # Labels untuk progress
    overall_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0', visibility='hidden'))
    step_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0', visibility='hidden'))
    
    # Container progress dengan styling yang lebih konsisten
    progress_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
        progress_bar,
        overall_label,
        current_progress,
        step_label
    ])
    
    # Status output dengan layout standar
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Log accordion dengan styling standar - terbuka secara default (selected_index=0)
    log_accordion = widgets.Accordion(children=[status], selected_index=0)
    log_accordion.set_title(0, f"{ICONS['file']} Preprocessing Logs")
    
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
    
    # Layout UI dengan divider standar
    from smartcash.ui.utils.layout_utils import create_divider
    
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
        progress_container,
        log_accordion,
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
        'split_selector': split_selector,
        'advanced_accordion': advanced_accordion,
        'preprocess_button': action_buttons['primary_button'],
        'preprocessing_button': action_buttons['primary_button'],  # Alias untuk kompatibilitas
        'stop_button': action_buttons['stop_button'],
        'reset_button': action_buttons['reset_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': action_buttons['save_button'],
        'button_container': action_buttons['container'],
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'overall_label': overall_label,
        'step_label': step_label,
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'module_name': 'preprocessing',
        # Tambahkan visualization_buttons untuk mencegah KeyError
        'visualization_buttons': visualization_buttons,
        # Default dataset paths
        'data_dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    }
    
    return ui_components
"""
File: smartcash/ui/dataset/augmentation_component.py
Deskripsi: Komponen UI untuk augmentasi dataset dengan tombol yang diseragamkan dan fokus pada data preprocessed
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk augmentasi dataset dengan fokus pada data preprocessed.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.info_boxes.augmentation_info import get_augmentation_info
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, BUTTON
    from smartcash.ui.utils.alert_utils import create_info_alert
    from smartcash.ui.helpers.action_buttons import create_action_buttons, create_visualization_buttons
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Menambah variasi data dari hasil preprocessing untuk balancing distribusi kelas")
    
    # Panel info status dengan komponen standar
    status_panel = widgets.HTML(
        value=create_info_alert("Konfigurasi augmentasi dataset dari data preprocessed", "info").value
    )
    
    # Augmentation options dengan struktur yang lebih ringkas dan slider workers
    aug_options = widgets.VBox([
        widgets.SelectMultiple(
            options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
            value=['Combined (Recommended)'],
            description='Types:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', height='100px')
        ),
        widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            description='Variations:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Text(
            value='aug',
            description='Prefix:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=True,
            description='Process bboxes',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Validate results',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=4,
            min=1,
            max=16,
            step=1,
            description='Workers:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        # Tambah checkbox untuk target balancing
        widgets.Checkbox(
            value=True,
            description='Balance classes',
            style={'description_width': 'initial'}
        )
    ])
    
    # Buat tombol aksi dengan komponen standar
    action_buttons = create_action_buttons(
        primary_label="Run Augmentation",
        primary_icon="random",
        cleanup_enabled=True
    )
    
    # Tombol visualisasi dengan komponen standar
    visualization_buttons = create_visualization_buttons()
    
    # Progress tracking dengan styling standar
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Overall:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Current:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    # Progress container dengan warna header yang terlihat
    progress_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
        progress_bar, 
        current_progress
    ])
    
    # Status output dengan layout standar
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[status], selected_index=None)
    log_accordion.set_title(0, f"{ICONS['file']} Augmentation Logs")
    
    # Visualization container
    visualization_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            min_height='100px',
            display='none'
        )
    )
    
    # Summary container
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            display='none'
        )
    )
    
    # Help panel dengan menggunakan info_box augmentation
    help_panel = get_augmentation_info(open_by_default=False)
    
    # Layout UI dengan divider standar
    from smartcash.ui.utils.layout_utils import create_divider
    
    # Container utama
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Augmentation Settings</h4>"),
        aug_options,
        create_divider(),
        action_buttons['container'],
        progress_container,
        log_accordion,
        summary_container,
        visualization_buttons['container'],
        visualization_container,
        help_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'aug_options': aug_options,
        'augment_button': action_buttons['primary_button'],
        'stop_button': action_buttons['stop_button'],
        'reset_button': action_buttons['reset_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': action_buttons['save_button'],
        'button_container': action_buttons['container'],
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'visualization_buttons': visualization_buttons['container'],
        'visualize_button': visualization_buttons['visualize_button'],
        'compare_button': visualization_buttons['compare_button'],
        'distribution_button': visualization_buttons['distribution_button'],
        'visualization_container': visualization_container,
        'module_name': 'augmentation'
    }
    
    return ui_components
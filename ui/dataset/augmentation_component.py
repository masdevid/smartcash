"""
File: smartcash/ui/dataset/augmentation_component.py
Deskripsi: Komponen UI untuk augmentasi dataset dengan tampilan yang lebih sederhana, slider workers dan perbaikan warna teks
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk augmentasi dataset dengan tampilan yang lebih sederhana.
    
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
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Menambah variasi data dengan augmentasi untuk meningkatkan performa model")
    
    # Panel info status dengan komponen standar
    status_panel = widgets.HTML(
        value=create_info_alert("Konfigurasi augmentasi dataset", "info").value
    )
    
    # Path setting dengan opsi untuk mengubah lokasi dataset
    location_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}; margin:5px 0'>{ICONS['folder']} Lokasi Dataset</h4>"),
        widgets.Text(
            value='data',
            description='Data Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', margin='5px 0')
        ),
        widgets.Text(
            value='data/augmented',
            description='Output Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', margin='5px 0')
        )
    ])
    
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
        )
    ])
    
    # Tombol-tombol dengan styling standar
    augment_button = widgets.Button(description='Run Augmentation', button_style='primary', icon='random')
    stop_button = widgets.Button(description='Stop', button_style='danger', icon='stop', 
                               layout=widgets.Layout(display='none'))
    reset_button = widgets.Button(description='Reset', button_style='warning', icon='refresh')
    cleanup_button = widgets.Button(description='Clean Augmented Data', button_style='danger', icon='trash')
    save_button = widgets.Button(description='Simpan Konfigurasi', button_style='success', icon='save')
    
    # Container tombol utama
    button_container = widgets.HBox([
        augment_button, stop_button, reset_button, save_button, cleanup_button
    ], layout=widgets.Layout(margin='10px 0', gap='10px'))
    
    # Progress tracking dengan styling standar
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Overall:',
                                    bar_style='info', orientation='horizontal',
                                    layout=widgets.Layout(visibility='hidden', width='100%'))
    current_progress = widgets.IntProgress(value=0, min=0, max=100, description='Current:',
                                         bar_style='info', orientation='horizontal',
                                         layout=widgets.Layout(visibility='hidden', width='100%'))
    
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
    
    # Visualization buttons
    visualization_buttons = widgets.HBox([
        widgets.Button(description='Tampilkan Sampel', button_style='info', icon='image'),
        widgets.Button(description='Bandingkan Hasil', button_style='info', icon='columns')
    ], layout=widgets.Layout(margin='10px 0', display='none', gap='10px'))
    
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
        location_container,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Augmentation Settings</h4>"),
        aug_options,
        create_divider(),
        button_container,
        progress_container,
        log_accordion,
        summary_container,
        visualization_buttons,
        visualization_container,
        help_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'data_dir_input': location_container.children[1],
        'output_dir_input': location_container.children[2],
        'aug_options': aug_options,
        'augment_button': augment_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'cleanup_button': cleanup_button,
        'save_button': save_button,
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'visualization_buttons': visualization_buttons,
        'visualization_container': visualization_container,
        'module_name': 'augmentation'
    }
    
    return ui_components
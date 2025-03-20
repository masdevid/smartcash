"""
File: smartcash/ui/dataset/augmentation_component.py
Deskripsi: Komponen UI untuk augmentasi dataset dengan integrasi UI utils standar dan perbaikan warna header
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk augmentasi dataset dengan memanfaatkan UI utils standar.
    
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
    
    # Augmentation options dengan struktur yang lebih ringkas
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
        widgets.Checkbox(
            value=True,
            description='Resume if interrupted',
            style={'description_width': 'initial'}
        )
    ])
    
    # Position parameters in accordion
    position_options = widgets.VBox([
        widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.05, 
            description='Flip prob:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.0, min=0, max=1, step=0.05, 
            description='Flipud prob:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.IntSlider(
            value=15, min=0, max=90, 
            description='Degrees:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.1, min=0, max=0.5, step=0.05, 
            description='Translate:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.1, min=0, max=0.5, step=0.05, 
            description='Scale:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.0, min=0, max=20, step=1.0, 
            description='Shear:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Lighting parameters in accordion
    lighting_options = widgets.VBox([
        widgets.FloatSlider(
            value=0.015, min=0, max=0.1, step=0.005, 
            description='HSV Hue:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.7, min=0, max=1, step=0.05, 
            description='HSV Sat:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.4, min=0, max=1, step=0.05, 
            description='HSV Value:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Contrast:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Brightness:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.0, min=0, max=1, step=0.05, 
            description='Compress:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Extreme options in accordion
    extreme_options = widgets.VBox([
        widgets.IntSlider(
            value=30, min=10, max=90, 
            description='Min rotation:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.IntSlider(
            value=90, min=30, max=180, 
            description='Max rotation:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Probability:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Tambahkan validation options yang lebih lengkap
    validation_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Check bounding boxes',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Fix truncated boxes',
            style={'description_width': 'initial'}
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Min visibility:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Advanced settings accordion
    advanced_accordion = widgets.Accordion(children=[position_options, lighting_options, extreme_options, validation_options])
    advanced_accordion.set_title(0, f"{ICONS['settings']} Position Parameters")
    advanced_accordion.set_title(1, f"{ICONS['settings']} Lighting Parameters")
    advanced_accordion.set_title(2, f"{ICONS['settings']} Extreme Parameters")
    advanced_accordion.set_title(3, f"{ICONS['settings']} Validation Parameters")
    advanced_accordion.selected_index = None  # Collapsed by default
    
    # Tombol-tombol preprocessing dengan styling standar
    augment_button = widgets.Button(description='Run Augmentation', button_style='primary', icon='random')
    stop_button = widgets.Button(description='Stop', button_style='danger', icon='stop', 
                               layout=widgets.Layout(display='none'))
    cleanup_button = widgets.Button(description='Clean Augmented Data', button_style='danger', icon='trash')
    save_button = widgets.Button(description='Simpan Konfigurasi', button_style='success', icon='save')
    
    # Container tombol utama
    button_container = widgets.HBox([
        augment_button, stop_button, save_button, cleanup_button
    ], layout=widgets.Layout(margin='10px 0', display='flex', justify_content='space-between'))
    
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
    ], layout=widgets.Layout(margin='10px 0', display='none'))
    
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
        advanced_accordion,
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
        'aug_options': aug_options,
        'position_options': position_options,
        'lighting_options': lighting_options,
        'extreme_options': extreme_options,
        'validation_options': validation_options,
        'advanced_accordion': advanced_accordion,
        'augment_button': augment_button,
        'stop_button': stop_button,
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
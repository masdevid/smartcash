"""
File: smartcash/ui/dataset/preprocessing_component.py
Deskripsi: Komponen UI untuk preprocessing dataset dengan antarmuka yang disederhanakan
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_preprocessing_ui(env=None, config=None) -> Dict[str, Any]:
    """Buat komponen UI untuk preprocessing dataset dengan antarmuka yang disederhanakan."""
    
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET, BUTTON

    # Header dengan komponen standar
    header = create_header(f"{ICONS['processing']} Dataset Preprocessing", 
                          "Preprocessing dataset untuk training model SmartCash")
    
    # Panel info status
    status_panel = widgets.HTML(
        value=f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Konfigurasi preprocessing dataset</p>
        </div>"""
    )
    
    # Preprocessing options dengan struktur yang lebih ringkas
    preprocess_options = widgets.VBox([
        widgets.IntSlider(value=640, min=320, max=640, step=32, description='Image size:',
                         style={'description_width': 'initial'}, layout=widgets.Layout(width='70%')),
        widgets.Checkbox(value=True, description='Enable normalization', style={'description_width': 'initial'}),
        widgets.Checkbox(value=True, description='Preserve aspect ratio', style={'description_width': 'initial'}),
        widgets.Checkbox(value=True, description='Enable caching', style={'description_width': 'initial'}),
        widgets.IntSlider(value=4, min=1, max=16, description='Workers:',
                         style={'description_width': 'initial'}, layout=widgets.Layout(width='50%'))
    ])
    
    # Validation options dalam accordion
    validation_options = widgets.VBox([
        widgets.Checkbox(value=True, description='Validate dataset integrity', style={'description_width': 'initial'}),
        widgets.Checkbox(value=True, description='Fix issues automatically', style={'description_width': 'initial'}),
        widgets.Checkbox(value=True, description='Move invalid files', style={'description_width': 'initial'}),
        widgets.Text(value='data/invalid', description='Invalid dir:', 
                    style={'description_width': 'initial'}, layout=widgets.Layout(width='60%'))
    ])
    
    # Accordion untuk validation options
    advanced_accordion = widgets.Accordion(children=[validation_options])
    advanced_accordion.set_title(0, f"{ICONS['search']} Validation Options")
    
    # Split selector untuk preprocessing
    split_selector = widgets.RadioButtons(
        options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
        value='All Splits', description='Process:', style={'description_width': 'initial'},
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Tombol-tombol preprocessing dengan styling standar
    preprocess_button = widgets.Button(description='Run Preprocessing', button_style='primary', icon='cog')
    stop_button = widgets.Button(description='Stop', button_style='danger', icon='stop', 
                               layout=widgets.Layout(display='none'))
    cleanup_button = widgets.Button(description='Clean Preprocessed Data', button_style='danger', 
                                  icon='trash', layout=widgets.Layout(display='none'))
    save_button = widgets.Button(description='Simpan Konfigurasi', button_style='success', icon='save')
    
    # Tombol-tombol visualisasi
    visualize_button = widgets.Button(description='Visualisasi Sampel', button_style='info',
                                    icon='image', layout=widgets.Layout(display='none'))
    compare_button = widgets.Button(description='Bandingkan Dataset', button_style='info',
                                  icon='columns', layout=widgets.Layout(display='none'))
    distribution_button = widgets.Button(description='Distribusi Kelas', button_style='info',
                                       icon='bar-chart', layout=widgets.Layout(display='none'))
    
    # Container tombol utama
    button_container = widgets.HBox([
        preprocess_button, stop_button, save_button, cleanup_button
    ], layout=widgets.Layout(margin='10px 0'))
    
    # Container tombol visualisasi
    visualization_buttons = widgets.HBox([
        visualize_button, compare_button, distribution_button
    ], layout=widgets.Layout(margin='10px 0', display='none'))
    
    # Progress tracking dengan styling standar
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Overall:',
                                    bar_style='info', orientation='horizontal',
                                    layout=widgets.Layout(visibility='hidden', width='100%'))
    current_progress = widgets.IntProgress(value=0, min=0, max=100, description='Current:',
                                        bar_style='info', orientation='horizontal',
                                        layout=widgets.Layout(visibility='hidden', width='100%'))
    
    progress_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
        progress_bar, 
        current_progress
    ])
    
    # Status output dengan layout standar
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Log accordion dengan styling standar
    log_accordion = widgets.Accordion(children=[status], selected_index=0)
    log_accordion.set_title(0, f"{ICONS['file']} Preprocessing Logs")
    
    # Summary stats container
    summary_container = widgets.Output(
        layout=widgets.Layout(border='1px solid #ddd', padding='10px', 
                             margin='10px 0', display='none')
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
        button_container,
        progress_container,
        log_accordion,
        summary_container,
        visualization_buttons,
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
        'preprocess_button': preprocess_button,
        'stop_button': stop_button,
        'cleanup_button': cleanup_button,
        'save_button': save_button,
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'visualization_buttons': visualization_buttons,
        'visualize_button': visualize_button,
        'compare_button': compare_button,
        'distribution_button': distribution_button,
        'module_name': 'preprocessing',
        # Default dataset paths
        'data_dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    }
    
    return ui_components
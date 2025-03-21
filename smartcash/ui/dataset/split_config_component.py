"""
File: smartcash/ui/dataset/split_config_component.py
Deskripsi: Komponen UI untuk konfigurasi pembagian dataset dengan tombol visualisasi on-demand
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.common.constants import DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH

def create_split_config_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi split dataset dengan visualisasi on-demand.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.alert_utils import create_info_box, create_info_alert
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.split_info import get_split_info
    
    # Deteksi status drive
    is_colab = 'google.colab' in str(globals())
    drive_mounted = False
    drive_path = None
    
    # Cek drive dari environment manager
    if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
        drive_mounted = True
        drive_path = str(env.drive_path) if hasattr(env, 'drive_path') else '/content/drive/MyDrive'
    
    # Header
    header = create_header(f"{ICONS['dataset']} Konfigurasi Split Dataset", 
                         "Konfigurasi dan visualisasi dataset untuk training, validation, dan testing")
    
    # Status panel
    status_panel = widgets.HTML(value=f"""
        <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                    color:{COLORS['alert_info_text']}; margin:10px 0; border-radius:4px; 
                    border-left:4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} {'Terhubung ke Google Drive 🟢' if drive_mounted else 'Google Drive tidak terhubung ⚪'}</p>
        </div>
    """)
    
    # Output box untuk visualisasi, status, dan log
    output_box = widgets.Output(
        layout=widgets.Layout(
            margin='10px 0',
            border='1px solid #ddd',
            padding='10px',
            min_height='200px',
            max_height='400px',
            overflow='auto'
        )
    )
    
    # Current statistics panel
    current_stats_html = widgets.HTML(
        value=f"""<div style="text-align:center; padding:15px;">
                <p style="color:{COLORS['muted']};">{ICONS['processing']} Memuat statistik dataset...</p>
               </div>""",
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Split percentages panel with sliders
    train_slider = widgets.FloatSlider(
        value=70.0, min=50.0, max=90.0, step=1.0,
        description='Train:',
        style={'description_width': '60px'},
        layout=widgets.Layout(width='70%'),
        readout_format='.0f'
    )
    
    valid_slider = widgets.FloatSlider(
        value=15.0, min=5.0, max=30.0, step=1.0,
        description='Valid:',
        style={'description_width': '60px'},
        layout=widgets.Layout(width='70%'),
        readout_format='.0f'
    )
    
    test_slider = widgets.FloatSlider(
        value=15.0, min=5.0, max=30.0, step=1.0,
        description='Test:',
        style={'description_width': '60px'},
        layout=widgets.Layout(width='70%'),
        readout_format='.0f'
    )
    
    stratified_checkbox = widgets.Checkbox(
        value=True,
        description='Stratified split (menjaga keseimbangan distribusi kelas)',
        style={'description_width': 'initial'}
    )
    
    split_panel = widgets.VBox([
        widgets.HTML(f"<h3 style='color:{COLORS['dark']}; margin-top:10px; margin-bottom:10px;'>{ICONS['folder']} Persentase Split</h3>"),
        train_slider, valid_slider, test_slider, stratified_checkbox
    ])
    
    # Advanced options panel
    advanced_options = widgets.VBox([
        widgets.IntText(
            value=42,
            description='Random seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=True,
            description='Backup dataset sebelum split',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='data/splits_backup',
            description='Direktori backup:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Pathways configuration
    data_paths = widgets.VBox([
        widgets.HTML(f"<h3 style='color:{COLORS['dark']}; margin-top:10px; margin-bottom:10px;'>{ICONS['folder']} Lokasi Dataset</h3>"),
        widgets.Text(
            value=DRIVE_DATASET_PATH if drive_mounted else 'data',
            description='Dataset Path:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value=DRIVE_PREPROCESSED_PATH if drive_mounted else 'data/preprocessed',
            description='Preprocessed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Advanced settings accordion
    advanced_accordion = widgets.Accordion(children=[advanced_options, data_paths], selected_index=None)
    advanced_accordion.set_title(0, f"{ICONS['settings']} Pengaturan Lanjutan")
    advanced_accordion.set_title(1, f"{ICONS['folder']} Lokasi Dataset")
    
    # Info notice
    split_notice = create_info_alert(
        f"Perlu diperhatikan bahwa split dataset sudah dilakukan melalui Roboflow. " +
        f"Pengaturan ini hanya digunakan untuk konfigurasi dan visualisasi distribusi dataset.",
        "info"
    )
    
    # Visualisasi button
    visualize_button = widgets.Button(
        description='Visualisasi Distribusi Kelas',
        button_style='info',
        icon='chart-bar',
        layout=widgets.Layout(margin='0 5px 0 0')
    )
    
    # Button container
    try:
        from smartcash.ui.training_config.config_buttons import create_config_buttons
        buttons_container = create_config_buttons("Split Dataset")
    except ImportError:
        # Fallback jika config_buttons tidak tersedia
        save_button = widgets.Button(
            description='Simpan Konfigurasi',
            button_style='primary',
            icon='save',
            layout=widgets.Layout(margin='0 5px')
        )
        
        reset_button = widgets.Button(
            description='Reset ke Default',
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(margin='0 0 0 5px')
        )
        
        buttons_container = widgets.HBox([save_button, reset_button], 
            layout=widgets.Layout(display='flex', justify_content='flex-end'))
    
    # Kombinasikan semua tombol dalam satu baris
    all_buttons = widgets.HBox([
        visualize_button,
        buttons_container
    ], layout=widgets.Layout(
        display='flex', 
        justify_content='space-between',
        margin='10px 0',
        align_items='center'
    ))
    
    # Info box
    info_box = get_split_info()
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        split_notice,
        current_stats_html,
        split_panel,
        advanced_accordion,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        all_buttons,
        output_box,
        info_box
    ])
    
    # Dictionary untuk akses komponen
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'output_box': output_box,
        'current_stats_html': current_stats_html,
        'split_panel': split_panel,
        'split_sliders': [train_slider, valid_slider, test_slider],
        'stratified': stratified_checkbox,
        'advanced_options': advanced_options,
        'data_paths': data_paths,
        'buttons_container': buttons_container,
        'visualize_button': visualize_button,
        'module_name': 'split_config'
    }
    
    # Pastikan buttons dapat diakses
    if isinstance(buttons_container, widgets.HBox):
        ui_components['save_button'] = buttons_container.children[0]
        ui_components['reset_button'] = buttons_container.children[1]
    else:
        # Extract buttons jika menggunakan create_config_buttons
        for child in buttons_container.children:
            if isinstance(child, widgets.Button):
                if child.description == 'Simpan Konfigurasi':
                    ui_components['save_button'] = child
                elif child.description == 'Reset ke Default':
                    ui_components['reset_button'] = child
    
    return ui_components
"""
File: smartcash/ui/dataset/split/components/split_component.py
Deskripsi: Komponen UI untuk konfigurasi pembagian dataset tanpa visualisasi
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
import os
from pathlib import Path
from smartcash.dataset.utils.dataset_constants import DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH, DEFAULT_PREPROCESSED_DIR

def create_split_component(env=None, config=None) -> Dict[str, Any]:
    """
    Membuat komponen UI untuk konfigurasi split dataset.
    
    Args:
        env: Environment dari notebook
        config: Konfigurasi dataset
        
    Returns:
        Dictionary berisi komponen UI
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
    
    # Header dengan style minimalis
    header = create_header(f"{ICONS['dataset']} Konfigurasi Split Dataset", 
                         "Konfigurasi pembagian dataset untuk training, validation, dan testing")
    
    # Status panel untuk pesan status
    status_panel = widgets.Output(
        layout=widgets.Layout(margin='5px 0', padding='0')
    )
    
    # Output box untuk status dan log
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
    
    # Status panel untuk menampilkan pesan status dan alert
    status_panel = widgets.Output(
        layout=widgets.Layout(
            margin='10px 0',
            min_height='50px'
        )
    )
    
    # Informasi konfigurasi panel
    config_info_html = widgets.HTML(
        value=f"""<div style="text-align:center; padding:15px;">
                <p style="color:{COLORS['muted']};">{ICONS['info']} Konfigurasi split dataset akan disimpan di dataset_config.yaml</p>
               </div>""",
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Split percentages panel
    # Slider untuk train split - dengan style compact
    train_slider = widgets.FloatSlider(
        value=0.7,
        min=0.5,
        max=0.9,
        step=0.05,
        description='Train:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Slider untuk validation split - dengan style compact
    val_slider = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.05,
        description='Val:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    # Slider untuk test split - dengan style compact
    test_slider = widgets.FloatSlider(
        value=0.15,
        min=0.05,
        max=0.3,
        step=0.05,
        description='Test:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    stratified_checkbox = widgets.Checkbox(
        value=True,
        description='Stratified split (menjaga keseimbangan distribusi kelas)',
        style={'description_width': 'initial'}
    )
    
    # Slider container dengan style compact
    slider_container = widgets.VBox([
        widgets.HTML(value=f"<h4 style='margin:5px 0;'>{ICONS['split']} Proporsi Split Dataset</h4>"),
        train_slider,
        val_slider,
        test_slider,
        widgets.HBox([stratified_checkbox], layout=widgets.Layout(margin='5px 0'))
    ], layout=widgets.Layout(margin='5px 0', padding='8px', 
                            border='1px solid #ddd', border_radius='5px'))
    
    # Advanced options panel
    advanced_options = widgets.VBox([
        widgets.IntText(
            value=42,
            description='Random Seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=True,
            description='Backup sebelum split',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='data/splits_backup',
            description='Backup Dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Data paths panel
    data_paths = widgets.VBox([
        widgets.Text(
            value=DRIVE_DATASET_PATH if drive_mounted else 'data',
            description='Dataset Path:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value=DRIVE_PREPROCESSED_PATH if drive_mounted else DEFAULT_PREPROCESSED_DIR,
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
        f"Perlu diperhatikan bahwa pengaturan ini hanya digunakan untuk memperbarui konfigurasi split di dataset_config.yaml " +
        f"dan tidak melakukan visualisasi atau pemrosesan data apapun.",
        "info"
    )
    
    # Buttons dengan style compact
    save_button = widgets.Button(
        description='Simpan',
        icon='save',
        button_style='primary',
        tooltip='Simpan konfigurasi split',
        layout=widgets.Layout(width='auto', margin='0 5px 0 0')
    )
    
    reset_button = widgets.Button(
        description='Reset',
        icon='refresh',
        button_style='warning',
        tooltip='Reset ke default',
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    # Buttons container dengan style compact
    buttons_container = widgets.HBox([save_button, reset_button], 
                                  layout=widgets.Layout(margin='5px 0', justify_content='flex-start', padding='0'))
    
    # Main container dengan style compact
    main_container = widgets.VBox([
        header,
        slider_container,
        buttons_container,
        status_panel,
        output_box
    ], layout=widgets.Layout(width='100%', padding='5px'))
    
    # Simpan komponen UI untuk digunakan di handlers
    ui_components = {
        'train_slider': train_slider,
        'val_slider': val_slider,
        'test_slider': test_slider,
        'stratified_checkbox': stratified_checkbox,
        'save_button': save_button,
        'reset_button': reset_button,
        'output_box': output_box,
        'status_panel': status_panel,
        'main_container': main_container
    }
    
    # Callback untuk memperbarui nilai slider saat salah satu slider berubah
    def update_sliders(change):
        # Pastikan total tetap 1.0
        total = train_slider.value + val_slider.value + test_slider.value
        if abs(total - 1.0) > 0.01:  # Toleransi untuk floating point
            # Sesuaikan nilai lain
            if change['owner'] == train_slider:
                # Distribusikan sisanya secara proporsional
                remaining = 1.0 - train_slider.value
                ratio = val_slider.value / (val_slider.value + test_slider.value) if (val_slider.value + test_slider.value) > 0 else 0.5
                val_slider.value = remaining * ratio
                test_slider.value = remaining * (1 - ratio)
            elif change['owner'] == val_slider:
                # Sesuaikan test slider
                test_slider.value = 1.0 - train_slider.value - val_slider.value
            elif change['owner'] == test_slider:
                # Sesuaikan valid slider
                val_slider.value = 1.0 - train_slider.value - test_slider.value
    
    # Tambahkan observers untuk slider
    train_slider.observe(update_sliders, names='value')
    val_slider.observe(update_sliders, names='value')
    test_slider.observe(update_sliders, names='value')
    
    # Simpan referensi ke split_sliders untuk cleanup
    ui_components['split_sliders'] = [train_slider, val_slider, test_slider]
    ui_components['ui'] = main_container
    
    return ui_components


def create_split_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Fungsi wrapper untuk kompatibilitas dengan kode lama.
    
    Args:
        env: Environment dari notebook
        config: Konfigurasi dataset
        
    Returns:
        Dictionary berisi komponen UI
    """
    return create_split_component(env, config)

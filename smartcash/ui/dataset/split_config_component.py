"""
File: smartcash/ui/dataset/split_config_component.py
Deskripsi: Komponen UI untuk konfigurasi pembagian dataset dengan visualisasi distribusi yang disederhanakan
"""

import ipywidgets as widgets
from typing import Dict, Any
import os
from pathlib import Path

def create_split_config_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi split dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI
    from smartcash.ui.components.headers import create_header
    from smartcash.ui.components.alerts import create_info_box, create_info_alert
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Coba load dataset config jika file ada
    dataset_config = {}
    try:
        import yaml
        dataset_config_path = 'configs/dataset_config.yaml'
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f) or {}
    except Exception:
        pass
    
    # Gabungkan dengan config utama
    if dataset_config and config and 'data' in config:
        config['data'].update(dataset_config.get('data', {}))
    elif dataset_config and 'data' in dataset_config:
        if not config:
            config = {}
        if 'data' not in config:
            config['data'] = {}
        config['data'].update(dataset_config.get('data', {}))
    
    # Header
    header = create_header(
        f"{ICONS['dataset']} Konfigurasi Split Dataset",
        "Konfigurasi pembagian dataset untuk training, validation, dan testing"
    )
    
    # Status panel
    status_panel = widgets.HTML(value=f"""
        <div style="padding: 10px; background-color: {COLORS['alert_info_bg']}; 
                    color: {COLORS['alert_info_text']}; margin: 10px 0; border-radius: 4px; 
                    border-left: 4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Pengaturan pembagian dataset untuk training, validation, dan testing</p>
        </div>
    """)
    
    # Unified output box untuk semua jenis output
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
    
    # Current statistics container
    current_stats_html = widgets.HTML(
        value=f"""<div style="text-align:center; padding:15px;">
                <p style="color:{COLORS['muted']};">{ICONS['processing']} Memuat statistik dataset...</p>
               </div>""",
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Split percentages panel with slider
    split_panel = widgets.VBox([
        widgets.HTML(f"<h3 style='color:{COLORS['dark']}; margin-top:10px; margin-bottom:10px;'>{ICONS['folder']} Persentase Split</h3>"),
        widgets.FloatSlider(
            value=70.0,
            min=50.0,
            max=90.0,
            step=1.0,
            description='Train:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='70%'),
            readout_format='.0f'
        ),
        widgets.FloatSlider(
            value=15.0,
            min=5.0,
            max=30.0,
            step=1.0,
            description='Valid:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='70%'),
            readout_format='.0f'
        ),
        widgets.FloatSlider(
            value=15.0,
            min=5.0,
            max=30.0,
            step=1.0,
            description='Test:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='70%'),
            readout_format='.0f'
        ),
        widgets.Checkbox(
            value=True,
            description='Stratified split (menjaga keseimbangan distribusi kelas)',
            style={'description_width': 'initial'}
        )
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
    
    # Advanced settings accordion
    advanced_accordion = widgets.Accordion(children=[advanced_options], selected_index=None)
    advanced_accordion.set_title(0, f"{ICONS['settings']} Pengaturan Lanjutan")
    
    # Notice about splits being done in Roboflow
    split_notice = create_info_alert(
        f"Perlu diperhatikan bahwa split dataset sudah dilakukan melalui Roboflow. " +
        f"Pengaturan ini hanya digunakan untuk konfigurasi dan visualisasi.",
        "info"
    )
    
    # Buttons container
    from smartcash.ui.training_config.config_buttons import create_config_buttons
    buttons_container = create_config_buttons("Split Dataset")
    
    # Info box with helpful information
    info_box = create_info_box(
        "Tentang Split Dataset",
        f"""
        <p>Pembagian dataset menjadi 3 subset:</p>
        <ul>
            <li><strong>Train</strong>: Dataset untuk pelatihan model (biasanya 70-80%)</li>
            <li><strong>Validation</strong>: Dataset untuk validasi selama pelatihan (biasanya 10-15%)</li>
            <li><strong>Test</strong>: Dataset untuk evaluasi akhir model (biasanya 10-15%)</li>
        </ul>
        <p>Gunakan <strong>stratified split</strong> untuk memastikan distribusi kelas tetap seimbang di semua subset.</p>
        <p>Total persentase harus 100%. Jika tidak, nilai akan disesuaikan secara otomatis.</p>
        """,
        'info',
        collapsed=True
    )
    
    # Rakit komponen UI dengan struktur baru - distribusi diatas slider
    ui = widgets.VBox([
        header,
        status_panel,
        split_notice,
        current_stats_html,  # Statistik distribusi diatas slider
        split_panel,
        advanced_accordion,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        buttons_container,
        output_box,  # Output box tunggal
        info_box
    ])
    
    # Dictionary untuk akses komponen dari luar
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'output_box': output_box,  # Output box tunggal untuk semua output
        'current_stats_html': current_stats_html,
        'split_panel': split_panel,
        'split_sliders': [split_panel.children[1], split_panel.children[2], split_panel.children[3]],
        'stratified': split_panel.children[4],
        'advanced_options': advanced_options,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'module_name': 'split_config'
    }
    
    return ui_components
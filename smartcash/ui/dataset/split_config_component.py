"""
File: smartcash/ui/dataset/split_config_component.py
Deskripsi: Komponen UI untuk konfigurasi pembagian dataset dengan dukungan Google Drive yang lebih terintegrasi dan ringan
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
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
    # Import komponen UI standar
    from smartcash.ui.components.headers import create_header
    from smartcash.ui.components.alerts import create_info_box, create_info_alert
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Deteksi drive dan status
    is_colab = 'google.colab' in str(globals())
    drive_mounted = False
    drive_path = None
    
    # Cek drive dari environment manager
    if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
        drive_mounted = True
        drive_path = str(env.drive_path) if hasattr(env, 'drive_path') else None
    
    # Header
    header = create_header(f"{ICONS['dataset']} Konfigurasi Split Dataset", 
                         "Konfigurasi pembagian dataset untuk training, validation, dan testing")
    
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
    
    # Current statistics container
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
    
    # Google Drive options panel jika diperlukan
    drive_options = None
    if is_colab or drive_mounted:
        use_drive_checkbox = widgets.Checkbox(
            value=drive_mounted,
            description='Gunakan Google Drive untuk dataset',
            style={'description_width': 'initial'},
            disabled=not drive_mounted
        )
        
        drive_path_input = widgets.Text(
            value='MyDrive/smartcash/data' if drive_mounted else '',
            description='Path ke data di drive:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%'),
            disabled=not drive_mounted
        )
        
        local_path_input = widgets.Text(
            value='data_local',
            description='Path lokal untuk clone:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%'),
            disabled=not drive_mounted
        )
        
        sync_checkbox = widgets.Checkbox(
            value=True,
            description='Sinkronisasi otomatis pada perubahan',
            style={'description_width': 'initial'},
            disabled=not drive_mounted
        )
        
        sync_button = widgets.Button(
            description='Sinkronisasi Sekarang',
            button_style='info',
            icon='sync',
            disabled=not drive_mounted,
            layout=widgets.Layout(width='auto', margin='10px 0')
        )
        
        drive_status_html = widgets.HTML(
            value=f"""<div style="padding:8px; margin-top:5px; background-color:{COLORS['alert_info_bg']}; 
                    color:{COLORS['alert_info_text']}; border-radius:4px; font-size:0.9em;">
                <p><strong>{ICONS['info']} Status Drive:</strong> {"Terhubung 🟢" if drive_mounted else "Tidak terhubung 🔴"}</p>
                {f"<p><strong>Path:</strong> {drive_path}</p>" if drive_mounted else ""}
                {f"<p>Aktifkan opsi ini untuk menggunakan dataset dari Google Drive dan clone ke lokal.</p>" if drive_mounted else
                 f"<p>Google Drive tidak terdeteksi. Gunakan <code>drive.mount('/content/drive')</code> untuk menghubungkan Drive.</p>"}
            </div>"""
        )
        
        drive_options = widgets.VBox([
            use_drive_checkbox, drive_path_input, local_path_input, 
            sync_checkbox, sync_button, drive_status_html
        ])
    
    # Advanced settings accordion with drive options
    accordion_children = [advanced_options]
    accordion_titles = [f"{ICONS['settings']} Pengaturan Lanjutan"]
    
    # Tambahkan opsi drive jika tersedia
    if drive_options:
        accordion_children.append(drive_options)
        accordion_titles.append(f"{ICONS['folder']} Opsi Google Drive")
    
    advanced_accordion = widgets.Accordion(children=accordion_children, selected_index=None)
    for i, title in enumerate(accordion_titles):
        advanced_accordion.set_title(i, title)
    
    # Notification
    split_notice = create_info_alert(
        f"Perlu diperhatikan bahwa split dataset sudah dilakukan melalui Roboflow. " +
        f"Pengaturan ini hanya digunakan untuk konfigurasi dan visualisasi.",
        "info"
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
            icon='save'
        )
        
        reset_button = widgets.Button(
            description='Reset ke Default',
            button_style='warning',
            icon='refresh'
        )
        
        buttons_container = widgets.HBox([save_button, reset_button])
    
    # Info box
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
        
        <h4>{ICONS['folder']} Opsi Google Drive</h4>
        <p>Jika dataset berada di Google Drive, aktifkan opsi Google Drive dan tentukan path ke dataset.</p>
        <p>Data akan di-clone ke path lokal untuk mempercepat akses dan menghindari kendala batas penggunaan Drive.</p>
        """,
        'info',
        collapsed=True
    )
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        split_notice,
        current_stats_html,
        split_panel,
        advanced_accordion,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        buttons_container,
        output_box,
        info_box
    ])
    
    # Pastikan buttons dapat diakses meskipun fallback
    if isinstance(buttons_container, widgets.HBox):
        save_button = buttons_container.children[0]
        reset_button = buttons_container.children[1]
    else:
        # Extract buttons dari komponen lain
        save_button = None
        reset_button = None
        for child in buttons_container.children:
            if isinstance(child, widgets.Button):
                if child.description == 'Simpan Konfigurasi':
                    save_button = child
                elif child.description == 'Reset ke Default':
                    reset_button = child
    
    # Dictionary untuk akses komponen dengan memastikan path yang tepat
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
        'save_button': save_button,
        'reset_button': reset_button,
        'module_name': 'split_config'
    }
    
    # Tambahkan komponen Drive jika ada
    if drive_options:
        ui_components['drive_options'] = drive_options
        ui_components['drive_sync_button'] = drive_options.children[4]  # sync_button
    
    return ui_components
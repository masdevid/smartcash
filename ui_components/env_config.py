"""
File: smartcash/ui_components/env_config.py
Author: Refactored
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash dengan desain modular.
"""

import ipywidgets as widgets
from IPython.display import HTML
from pathlib import Path

def create_env_config_ui():
    """
    Buat komponen UI untuk konfigurasi environment SmartCash.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("""
    <div style="background-color: #f0f8ff; padding: 15px; color: black; 
              border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #3498db;">
        <h2 style="color: inherit; margin-top: 0;">⚙️ Environment Configuration</h2>
        <p style="color: inherit; margin-bottom: 0;">Konfigurasi lingkungan kerja SmartCash</p>
    </div>
    """)
    
    # Colab info panel
    colab_panel = widgets.HTML("")
    
    # Environment info panel
    info_panel = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            max_height='200px',
            overflow='auto'
        )
    )
    
    # Button container
    button_container = widgets.HBox(layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        margin='10px 0'
    ))
    
    # Drive connection button
    drive_button = widgets.Button(
        description='Connect Google Drive',
        button_style='info',
        icon='link',
        tooltip='Mount and connect to Google Drive',
        layout=widgets.Layout(
            margin='0 10px 0 0',
            display='none'  # Hidden by default
        )
    )
    
    # Directory setup button
    dir_button = widgets.Button(
        description='Setup Directory Structure',
        button_style='primary',
        icon='folder-plus',
        tooltip='Create project directory structure',
        layout=widgets.Layout(margin='0')
    )
    
    # Add buttons to container
    button_container.children = [drive_button, dir_button]
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Info box
    info_box = widgets.HTML("""
    <div style="padding: 10px; background-color: #d1ecf1; border-left: 4px solid #0c5460; 
             color: #0c5460; margin: 10px 0; border-radius: 4px;">
        <h4 style="margin-top: 0; color: inherit;">ℹ️ Environment Setup</h4>
        <p>SmartCash mendukung dua environment kerja:</p>
        <ul>
            <li><strong>Google Colab</strong>: Dengan integrasi Google Drive untuk penyimpanan</li>
            <li><strong>Local</strong>: Untuk pengembangan dan evaluasi di mesin lokal</li>
        </ul>
        <p>Struktur direktori yang akan dibuat:</p>
        <pre style="margin: 0 0 0 10px; color: #0c5460; background: transparent; border: none;">
data/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
configs/
runs/train/weights/
logs/
exports/
        </pre>
    </div>
    """)
    
    # Assemble UI
    main_container.children = [
        header,
        colab_panel,
        info_panel,
        button_container,
        status,
        info_box
    ]
    
    # Dictionary untuk akses komponen dari luar
    ui_components = {
        'ui': main_container,
        'colab_panel': colab_panel,
        'info_panel': info_panel,
        'drive_button': drive_button,
        'dir_button': dir_button,
        'status': status
    }
    
    return ui_components
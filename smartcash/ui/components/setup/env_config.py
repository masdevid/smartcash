"""
File: smartcash/ui/components/setup/env_config.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash di subdirektori setup
"""

import ipywidgets as widgets
from IPython.display import HTML

from smartcash.ui.components.shared.headers import create_header
from smartcash.ui.components.shared.alerts import create_info_box

def create_env_config_ui(env=None, config=None):
    """
    Buat komponen UI untuk konfigurasi environment SmartCash.
    
    Args:
        env: Environment manager (optional)
        config: Konfigurasi (optional)
        
    Returns:
        Dict berisi widget UI dan referensi ke komponen utama
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_header(
        "Environment Configuration",
        "Konfigurasi lingkungan kerja SmartCash",
        "⚙️"
    )
    
    # Colab info panel
    colab_panel = widgets.HTML("")
    
    # Environment info panel (collapsed by default)
    info_panel = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            max_height='200px',
            overflow='auto'
        )
    )
    
    # Wrap in accordion to make collapsible
    info_accordion = widgets.Accordion(children=[info_panel], selected_index=None)
    info_accordion.set_title(0, "🖥️ System Information")
    
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
    
    # Help info box
    help_info = create_info_box(
        "Tentang Environment Setup",
        """
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
        """,
        'info',
        collapsed=True
    )
    
    # Assemble UI
    main_container.children = [
        header,
        colab_panel,
        help_info,
        button_container,
        status,
        info_accordion
    ]
    
    # Dictionary untuk akses komponen dari luar
    ui_components = {
        'ui': main_container,
        'colab_panel': colab_panel,
        'info_panel': info_panel,
        'info_accordion': info_accordion,
        'drive_button': drive_button,
        'dir_button': dir_button,
        'status': status
    }
    
    return ui_components
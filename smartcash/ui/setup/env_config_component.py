"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash dengan styling default widget
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config dengan styling default widget.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI
    from smartcash.ui.components.headers import create_header
    from smartcash.ui.components.alerts import create_info_box
    from smartcash.ui.utils.constants import ICONS
    
    # Header
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment untuk project SmartCash"
    )
    
    # Panel status Colab/Drive
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Panel info bantuan
    help_box = create_info_box(
        "Informasi Konfigurasi", 
        """<p>Konfigurasi environment akan memastikan project SmartCash berjalan dengan baik di lingkungan saat ini.</p>
        <ul>
            <li>Di Google Colab: Sebaiknya hubungkan ke Google Drive untuk menyimpan dataset dan model</li>
            <li>Di lingkungan lokal: Pastikan struktur direktori telah dibuat</li>
        </ul>""",
        style="info",
        collapsed=True
    )
    
    # Tombol aksi dengan styling default
    drive_button = widgets.Button(
        description='Hubungkan Google Drive',
        button_style='primary',
        icon='link',
        tooltip='Mount Google Drive dan siapkan struktur direktori'
    )
    
    directory_button = widgets.Button(
        description='Setup Direktori Lokal',
        button_style='info',
        icon='folder-plus',
        tooltip='Buat struktur direktori lokal'
    )
    
    # Kelompokkan tombol dalam grup horizontal dengan spacing
    button_group = widgets.HBox(
        [drive_button, directory_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            gap='10px',
            margin='10px 0'
        )
    )
    
    # Panel output
    status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Divider
    divider = widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>")
    
    # Container utama
    ui = widgets.VBox(
        [header, colab_panel, help_box, divider, button_group, status],
        layout=widgets.Layout(width='100%', padding='10px')
    )
    
    # Komponen UI
    ui_components = {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'help_panel': help_box,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'status': status
    }
    
    return ui_components
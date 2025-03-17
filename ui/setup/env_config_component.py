"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash dengan styling yang lebih baik
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config dengan tema konsisten.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI yang sudah ada
    from smartcash.ui.components.headers import create_header
    from smartcash.ui.components.alerts import create_info_box
    from smartcash.ui.components.widget_layouts import (
        main_container, button, section_container, output_area, 
        create_divider, BUTTON_LAYOUTS, GROUP_LAYOUTS
    )
    from smartcash.ui.utils.constants import ICONS, FA_ICONS
    
    # Header dengan styling konsisten
    header = create_header(
        "⚙️ Konfigurasi Environment", 
        "Setup environment untuk project SmartCash"
    )
    
    # Panel status Colab/Drive dengan komponen yang sudah ada
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Panel info bantuan dengan komponen yang sudah ada
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
    
    # Tombol aksi dengan styling konsisten dan icon yang benar
    drive_button = widgets.Button(
        description='Hubungkan Google Drive',
        button_style='primary',
        icon=FA_ICONS['link'],
        tooltip='Mount Google Drive dan siapkan struktur direktori',
        layout=BUTTON_LAYOUTS['standard']
    )
    
    directory_button = widgets.Button(
        description='Setup Direktori Lokal',
        button_style='info',
        icon=FA_ICONS['folder-plus'],
        tooltip='Buat struktur direktori lokal',
        layout=BUTTON_LAYOUTS['standard']
    )
    
    # Kelompokkan tombol dalam grup horizontal
    button_group = widgets.GridBox(
        [drive_button, directory_button],
        layout=GROUP_LAYOUTS['horizontal']
    )
    
    # Panel output dengan komponen yang sudah ada
    status = widgets.Output(layout=output_area)
    
    # Divider
    divider = create_divider()
    
    # Container utama dengan styling konsisten
    ui = widgets.VBox([
        header,
        colab_panel,
        help_box,
        divider,
        button_group,
        status
    ], layout=main_container)
    
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
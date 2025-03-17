"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_env_config_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk environment config.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Header
    header = widgets.HTML(
        """<div style="background:#f0f8ff; padding:15px; border-radius:5px; border-left:5px solid #3498db; margin-bottom:15px">
            <h1 style="margin:0; color:#2c3e50">⚙️ Konfigurasi Environment</h1>
            <p style="margin:5px 0; color:#7f8c8d">Setup environment untuk project SmartCash</p>
        </div>"""
    )
    
    # Panel status Colab/Drive
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Panel info bantuan
    help_panel = widgets.HTML(
        """<div style="padding:10px; background-color:#f8f9fa; border-left:4px solid #6c757d; margin:10px 0">
            <h3>ℹ️ Informasi</h3>
            <p>Konfigurasi environment akan memastikan project SmartCash berjalan dengan baik di lingkungan saat ini.</p>
            <ul>
                <li>Di Google Colab: Sebaiknya hubungkan ke Google Drive untuk menyimpan dataset dan model</li>
                <li>Di lingkungan lokal: Pastikan struktur direktori telah dibuat</li>
            </ul>
        </div>"""
    )
    
    # Tombol aksi
    drive_button = widgets.Button(
        description='Hubungkan Google Drive',
        button_style='primary',
        icon='link',
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    directory_button = widgets.Button(
        description='Setup Direktori Lokal',
        button_style='info',
        icon='folder-plus',
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Panel output
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            min_height='150px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Container utama
    ui = widgets.VBox([
        header,
        colab_panel,
        help_panel,
        drive_button,
        directory_button,
        status
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Komponen UI
    ui_components = {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'help_panel': help_panel,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'status': status
    }
    
    return ui_components
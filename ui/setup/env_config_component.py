"""
File: smartcash/ui/01_setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash dengan koneksi Google Drive dan setup direktori
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any

from smartcash.ui.components.headers import create_header
from smartcash.ui.components.alerts import create_info_alert
from smartcash.ui.components.layouts import STANDARD_LAYOUTS, MAIN_CONTAINER, OUTPUT_WIDGET, BUTTON

def create_env_config_ui(env: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi environment.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Header
    header = create_header(
        title="🛠️ Konfigurasi Environment",
        description="Setup lingkungan kerja SmartCash dengan koneksi Google Drive dan struktur direktori",
    )
    
    # Panel info Colab/Local
    colab_panel = widgets.HTML(value="Mendeteksi environment...")
    
    # Informasi sistem
    info_panel = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Tombol koneksi drive
    drive_button = widgets.Button(
        description='🔗 Hubungkan Google Drive',
        button_style='info',
        layout=BUTTON
    )
    
    # Tombol setup direktori
    dir_button = widgets.Button(
        description='📁 Setup Direktori Proyek',
        button_style='primary',
        layout=BUTTON
    )
    
    # Output untuk status dan hasil
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Output untuk tree direktori
    dir_output = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Container utama
    ui = widgets.VBox([
        header,
        colab_panel,
        info_panel,
        widgets.HBox([drive_button, dir_button], layout=STANDARD_LAYOUTS['hbox']),
        progress_bar,
        status,
        dir_output
    ], layout=MAIN_CONTAINER)
    
    # Return komponen
    return {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'info_panel': info_panel,
        'drive_button': drive_button,
        'dir_button': dir_button,
        'status': status,
        'dir_output': dir_output,
        'progress_bar': progress_bar,
        'config': config,
        'env': env,
        'module_name': 'env_config'
    }
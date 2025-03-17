"""
File: smartcash/ui/setup/env_config_component.py
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash
"""

import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.components.headers import create_header
from smartcash.ui.components.layouts import (
    STANDARD_LAYOUTS, 
    OUTPUT_WIDGET,
    HORIZONTAL_GROUP
)
from smartcash.ui.utils.constants import COLORS, ICONS

def create_env_config_ui(env=None, config=None):
    """
    Buat UI untuk konfigurasi environment.
    
    Args:
        env: Environment manager (opsional)
        config: Konfigurasi environment (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    # Header utama
    header = create_header(
        "🖥️ Konfigurasi Environment", 
        "Setup lingkungan proyek SmartCash"
    )
    
    # Panel informasi Colab
    colab_panel = widgets.HTML(value="", layout=widgets.Layout(width='100%'))
    
    # Panel informasi sistem
    info_panel = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Status output
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Tombol koneksi Drive (hanya di Colab)
    drive_button = widgets.Button(
        description='Hubungkan Google Drive',
        icon='cloud-upload',
        button_style='primary',
        layout=HORIZONTAL_GROUP
    )
    
    # Tombol setup direktori lokal
    dir_button = widgets.Button(
        description='Setup Direktori Lokal',
        icon='folder-open',
        button_style='success',
        layout=HORIZONTAL_GROUP
    )
    
    # Buat help info yang collapsible
    help_content = """
    <h4>🛠️ Petunjuk Konfigurasi Environment</h4>
    <ul>
        <li><strong>Google Colab</strong>: Gunakan tombol 'Hubungkan Google Drive' untuk menyambungkan dan mensinkronkan proyek</li>
        <li><strong>Local Environment</strong>: Gunakan tombol 'Setup Direktori Lokal' untuk membuat struktur direktori proyek</li>
        <li>Pastikan anda sudah clone repository SmartCash sebelum konfigurasi</li>
    </ul>
    """
    help_box = widgets.Accordion(children=[widgets.HTML(help_content)])
    help_box.set_title(0, "📖 Panduan Konfigurasi")
    help_box.selected_index = None  # Collapsible by default
    
    # Susun layout UI
    ui = widgets.VBox([
        header,
        colab_panel,
        info_panel,
        status,
        help_box,
        widgets.HBox([drive_button, dir_button])
    ])
    
    # Kembalikan komponen UI dalam dictionary
    return {
        'ui': ui,
        'header': header,
        'colab_panel': colab_panel,
        'info_panel': info_panel,
        'status': status,
        'drive_button': drive_button,
        'dir_button': dir_button,
        'help_box': help_box
    }
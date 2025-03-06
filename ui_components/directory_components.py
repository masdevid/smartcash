"""
File: smartcash/ui_components/directory_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk setup dan manajemen direktori project, termasuk integrasi dengan Google Drive.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

def create_directory_ui():
    """
    Buat UI komponen untuk setup dan manajemen direktori.
    
    Returns:
        Dictionary berisi komponen UI
    """
    # Header dan deskripsi
    header = widgets.HTML("<h2>📁 Setup Direktori</h2>")
    description = widgets.HTML(
        "<p>Setup direktori project dengan integrasi Google Drive.</p>"
    )
    
    # Google Drive integration
    drive_checkbox = widgets.Checkbox(
        value=True,
        description='Gunakan Google Drive untuk penyimpanan',
        disabled=False,
        indent=False
    )
    
    drive_path_text = widgets.Text(
        value='/content/drive/MyDrive/SmartCash',
        placeholder='Path ke direktori Google Drive',
        description='Path Drive:',
        disabled=False,
        layout=widgets.Layout(width='70%')
    )
    
    # Status indicator
    status_indicator = widgets.HTML(
        value="<p>Status: <span style='color: gray'>Belum disetup</span></p>"
    )
    
    # Tombol setup
    setup_button = widgets.Button(
        description='Setup Direktori',
        button_style='primary',
        icon='folder-plus',
        tooltip='Setup direktori project di lokasi yang dipilih'
    )
    
    # Output area
    output_area = widgets.Output()
    
    # Directory tree display
    directory_tree = widgets.HTML(
        value="<p><i>Directory tree akan ditampilkan setelah setup</i></p>"
    )
    
    # Susun layout UI
    drive_box = widgets.VBox([
        widgets.HTML("<h3>🔄 Integrasi Google Drive</h3>"),
        drive_checkbox,
        drive_path_text
    ])
    
    status_box = widgets.VBox([
        widgets.HTML("<h3>📊 Status Direktori</h3>"),
        status_indicator
    ])
    
    # Gabungkan komponen dalam layout utama
    ui_container = widgets.VBox([
        header,
        description,
        drive_box,
        status_box,
        setup_button,
        output_area,
        widgets.HTML("<h3>🌲 Struktur Direktori</h3>"),
        directory_tree
    ])
    
    # Return komponen untuk digunakan di handler
    return {
        'ui': ui_container,
        'drive_checkbox': drive_checkbox,
        'drive_path_text': drive_path_text,
        'status_indicator': status_indicator,
        'setup_button': setup_button,
        'output_area': output_area,
        'directory_tree': directory_tree
    }
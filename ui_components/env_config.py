"""
File: smartcash/ui_components/env_config.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk konfigurasi environment SmartCash
"""

import ipywidgets as widgets
from IPython.display import HTML
from pathlib import Path

def create_env_config_ui():
    """Buat UI konfigurasi environment SmartCash"""
    main = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("<h1>⚙️ Setup Environment</h1><p>Konfigurasi lingkungan kerja SmartCash</p>")
    
    # Colab info panel
    colab_panel = widgets.HTML("")
    
    # Google Drive button
    drive_btn = widgets.Button(
        description='Connect Google Drive',
        button_style='info',
        icon='link',
        tooltip='Connect to Google Drive',
        layout={'margin': '10px 0', 'display': 'none'}
    )
    
    # Directory structure button
    dir_btn = widgets.Button(
        description='Setup Directory Structure',
        button_style='primary',
        icon='folder-plus',
        tooltip='Create necessary directories',
        layout={'margin': '10px 0'}
    )
    
    # Status output
    status = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '100px', 'margin': '10px 0'})
    
    # Help accordion
    help_info = widgets.Accordion(children=[widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Environment Setup</h4>
            <ol>
                <li><b>Colab:</b> Klik tombol "Connect Google Drive" untuk menghubungkan ke Google Drive</li>
                <li><b>Directory:</b> Klik "Setup Directory Structure" untuk membuat struktur direktori project</li>
            </ol>
            <p>Struktur direktori yang akan dibuat:</p>
            <ul>
                <li><code>data/</code> - Dataset training, validasi, dan testing</li>
                <li><code>models/</code> - Model yang diexport</li>
                <li><code>runs/</code> - Hasil training dan deteksi</li>
                <li><code>configs/</code> - File konfigurasi</li>
                <li><code>logs/</code> - Log proses</li>
                <li><code>results/</code> - Hasil evaluasi dan visualisasi</li>
            </ul>
        </div>
    """)], selected_index=None)
    
    help_info.set_title(0, "ℹ️ Bantuan")
    
    # Assemble UI
    main.children = [
        header,
        colab_panel,
        drive_btn,
        dir_btn,
        status,
        help_info
    ]
    
    # Return components
    return {
        'ui': main,
        'colab_panel': colab_panel,
        'drive_button': drive_btn,
        'dir_button': dir_btn,
        'status': status
    }
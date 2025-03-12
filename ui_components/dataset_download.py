"""
File: smartcash/ui_components/dataset_download.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk download dataset SmartCash dengan pendekatan modular.
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import create_component_header, create_info_box

def create_dataset_download_ui():
    """
    Buat komponen UI untuk download dataset.
    
    Returns:
        Dict berisi widget UI dan referensi ke komponen utamanya
    """
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Dataset Download",
        "Download dataset untuk training model SmartCash",
        "📊"
    )
    
    # Download options
    download_options = widgets.RadioButtons(
        options=['Roboflow (Online)', 'Local Data (Upload)'],
        description='Source:',
        style={'description_width': 'initial'},
    )
    
    # Roboflow settings
    roboflow_settings = widgets.VBox([
        widgets.Password(
            value='',
            description='API Key:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value='smartcash-wo2us',
            description='Workspace:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value='rupiah-emisi-2022',
            description='Project:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Text(
            value='3',
            description='Version:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        )
    ])
    
    # Local upload widget
    local_upload = widgets.VBox([
        widgets.FileUpload(
            description='Upload ZIP:',
            accept='.zip',
            multiple=False,
            layout=widgets.Layout(width='300px')
        ),
        widgets.Text(
            value='data/uploaded',
            description='Target dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        )
    ])
    
    # Kondisional menampilkan settings berdasarkan pilihan
    download_settings_container = widgets.VBox([roboflow_settings])
    
    # Download button
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download',
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Status dan progress
    download_status = widgets.Output(
        layout=widgets.Layout(width='100%', border='1px solid #ddd', min_height='100px', margin='10px 0')
    )
    
    download_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(margin='10px 0', width='100%')
    )
    
    # Help accordion
    help_info = create_info_box(
        "Panduan Download Dataset",
        """
        <h4>Cara Download Dataset</h4>
        <ol>
            <li><b>Roboflow:</b> Masukkan API key Roboflow dan konfigurasi workspace/project</li>
            <li><b>Local:</b> Upload file ZIP yang berisi dataset dalam format YOLOv5</li>
        </ol>
        <h4>Setup API Key Roboflow</h4>
        <p>Anda memiliki beberapa cara untuk menyediakan API key Roboflow:</p>
        <ol>
            <li><b>Google Secret (untuk Colab):</b>
                <ul>
                    <li>Klik ikon kunci 🔑 di sidebar kiri</li>
                    <li>Tambahkan secret baru dengan nama <code>ROBOFLOW_API_KEY</code></li>
                    <li>Masukkan API key Roboflow Anda sebagai nilai</li>
                    <li>Klik "Save". API key akan diambil otomatis</li>
                </ul>
            </li>
            <li><b>Input Manual:</b> Masukkan API key secara langsung pada field yang tersedia</li>
            <li><b>Config File:</b> Tambahkan ke <code>configs/base_config.yaml</code></li>
        </ol>
        <h4>Struktur Dataset YOLOv5</h4>
        <pre>
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
        </pre>
        """,
        'info',
        icon="ℹ️",
        collapsed=True
    )
    
    # Rakit komponen UI
    main_container.children = [
        header,
        download_options,
        download_settings_container,
        download_button,
        download_progress,
        download_status,
        help_info
    ]
    
    # Return komponen UI
    ui_components = {
        'ui': main_container,
        'download_options': download_options,
        'roboflow_settings': roboflow_settings,
        'local_upload': local_upload,
        'download_settings_container': download_settings_container,
        'download_button': download_button,
        'download_progress': download_progress,
        'download_status': download_status,
    }
    
    return ui_components
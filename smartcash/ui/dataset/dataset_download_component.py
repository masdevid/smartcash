"""
File: smartcash/ui/dataset/dataset_download_component.py
Deskripsi: Komponen UI untuk download dataset SmartCash dengan pendekatan modular yang konsisten dan opsi backup
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_dataset_download_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk download dataset dengan design pattern yang konsisten.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.alert_utils import create_info_box
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Header
    header = create_header(
        f"{ICONS['dataset']} Download Dataset",
        "Download dan persiapan dataset untuk SmartCash"
    )
    
    # Panel info status
    status_panel = widgets.HTML(value=f"""
        <div style="padding: 10px; background-color: {COLORS['alert_info_bg']}; 
                    color: {COLORS['alert_info_text']}; margin: 10px 0; border-radius: 4px; 
                    border-left: 4px solid {COLORS['alert_info_text']};">
            <p style="margin:5px 0">{ICONS['info']} Pilih sumber dataset dan konfigurasi download</p>
        </div>
    """)
    
    # Download options
    download_options = widgets.RadioButtons(
        options=['Roboflow (Online)', 'Local Data (Upload)'],
        description='Sumber:',
        style={'description_width': 'initial'}
    )
    
    # Roboflow settings with backup checkbox
    roboflow_settings = widgets.VBox([
        widgets.Password(
            value='',
            description='API Key:',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='smartcash-wo2us',
            description='Workspace:',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='rupiah-emisi-2022',
            description='Project:',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='3',
            description='Version:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=False,
            description='Backup dataset lama (opsional)',
            indent=False
        )
    ])
    
    # Local upload widget with backup checkbox
    local_upload = widgets.VBox([
        widgets.FileUpload(
            description='Upload ZIP:',
            accept='.zip',
            multiple=False
        ),
        widgets.Text(
            value='data/uploaded',
            description='Target dir:',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=False,
            description='Backup dataset lama (opsional)',
            indent=False
        )
    ])
    
    # Tombol aksi
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download'
    )
    
    # Progress bar - Ubah ke full width
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Download: 0%',
        bar_style='',
        orientation='horizontal',
        style={'margin-top': '10px'},
        layout=widgets.Layout(width='75%'))
    
    # Label untuk progress
    progress_label = widgets.Label('Siap untuk download dataset')
    # Status output
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
    
    # Panel info bantuan
    help_panel = create_info_box(
        f"Panduan Download Dataset",
        f"""
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
                    <li>Klik ikon kunci {ICONS['settings']} di sidebar kiri</li>
                    <li>Tambahkan secret baru dengan nama <code>ROBOFLOW_API_KEY</code></li>
                    <li>Masukkan API key Roboflow Anda sebagai nilai</li>
                    <li>Klik "Save". API key akan diambil otomatis</li>
                </ul>
            </li>
            <li><b>Input Manual:</b> Masukkan API key secara langsung pada field yang tersedia</li>
            <li><b>Config File:</b> Tambahkan ke <code>configs/base_config.yaml</code></li>
        </ol>
        <h4>Backup Dataset</h4>
        <p>Jika opsi backup diaktifkan, dataset yang ada akan dibackup ke file ZIP sebelum diubah. File ZIP akan disimpan di direktori <code>data/backups</code>.</p>
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
        collapsed=True
    )
    
    # Container untuk download settings (akan berubah berdasarkan pilihan)
    download_settings_container = widgets.VBox([roboflow_settings])
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        download_options,
        download_settings_container,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        download_button,
        widgets.HBox([progress_bar, progress_label], layout=widgets.Layout(width='100%', margin='10px 0')),
        status,
        help_panel
    ])
    
    # Komponen UI
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'download_options': download_options,
        'roboflow_settings': roboflow_settings,
        'local_upload': local_upload,
        'download_settings_container': download_settings_container,
        'download_button': download_button,
        'progress_bar': progress_bar,
        'progress_label': progress_label,
        'status': status,
        'status_output': status,  # Alias untuk kompatibilitas
        'help_panel': help_panel,
        'module_name': 'dataset_download'
    }
    
    return ui_components
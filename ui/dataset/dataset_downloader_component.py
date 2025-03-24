"""
File: smartcash/ui/dataset/dataset_downloader_component.py
Deskripsi: Komponen UI untuk download dataset dengan area konfirmasi dan panel status yang lebih terlihat
"""

import ipywidgets as widgets
import os
from typing import Dict, Any, Optional

def create_dataset_downloader_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk download dataset dengan dukungan API key dari secrets.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.dataset_info import get_dataset_info
    
    # Header
    header = create_header(
        "📥 Dataset Downloader", 
        "Download dataset untuk training dan validation"
    )
    
    # Endpoint selection
    endpoint_label = widgets.HTML(f"<b>{ICONS['data']} Pilih Endpoint Dataset</b>")
    endpoint_dropdown = widgets.Dropdown(
        options=['Roboflow', 'Google Drive', 'URL Kustom'],
        value='Roboflow',
        description='Sumber:',
        layout=widgets.Layout(width='40%')
    )
    
    endpoint_container = widgets.VBox(
        [endpoint_label, endpoint_dropdown],
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Ambil nilai default dari konfigurasi
    roboflow_config = config.get('data', {}).get('roboflow', {})
    rf_workspace_default = roboflow_config.get('workspace', '')
    rf_project_default = roboflow_config.get('project', '')
    rf_version_default = roboflow_config.get('version', '')
    
    # Coba dapatkan API key dari environment variables
    api_key_env = os.environ.get('ROBOFLOW_API_KEY', '')
    
    # Roboflow Config
    rf_workspace = widgets.Text(
        value=rf_workspace_default,
        placeholder='Workspace ID',
        description='Workspace:',
        layout=widgets.Layout(width='80%')
    )
    
    rf_project = widgets.Text(
        value=rf_project_default,
        placeholder='Project ID',
        description='Project:',
        layout=widgets.Layout(width='80%')
    )
    
    rf_version = widgets.Text(
        value=rf_version_default,
        placeholder='Version',
        description='Version:',
        layout=widgets.Layout(width='40%')
    )
    
    rf_apikey = widgets.Password(
        value=api_key_env,
        placeholder='API Key',
        description='API Key:',
        layout=widgets.Layout(width='80%')
    )
    
    # Layout roboflow dalam accordion untuk kolapsibilitas
    rf_container = widgets.VBox(
        [rf_workspace, rf_project, rf_version, rf_apikey],
        layout=widgets.Layout(margin='5px 0')
    )
    
    rf_accordion = widgets.Accordion(
        [rf_container],
        selected_index=0,
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    rf_accordion.set_title(0, f"{ICONS['config']} Konfigurasi Roboflow")
    
    # Google Drive Config
    drive_folder = widgets.Text(
        value=config.get('data', {}).get('drive_folder', 'SmartCash/datasets'),
        placeholder='Path folder di Drive',
        description='Folder:',
        layout=widgets.Layout(width='80%')
    )
    
    # Layout Drive dalam accordion
    drive_container = widgets.VBox(
        [drive_folder],
        layout=widgets.Layout(margin='5px 0')
    )
    
    drive_accordion = widgets.Accordion(
        [drive_container],
        selected_index=None,  # Collapsed by default
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    drive_accordion.set_title(0, f"{ICONS['folder']} Konfigurasi Google Drive")
    
    # URL Kustom Config
    url_input = widgets.Text(
        value='',
        placeholder='https://example.com/dataset.zip',
        description='URL:',
        layout=widgets.Layout(width='80%')
    )
    
    # Layout URL dalam accordion
    url_container = widgets.VBox(
        [url_input],
        layout=widgets.Layout(margin='5px 0')
    )
    
    url_accordion = widgets.Accordion(
        [url_container],
        selected_index=None,  # Collapsed by default
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    url_accordion.set_title(0, f"{ICONS['link']} Konfigurasi URL")
    
    # Output lokasi dan format
    output_label = widgets.HTML(f"<b>{ICONS['folder']} Output Dataset</b>")
    
    output_format = widgets.Dropdown(
        options=['YOLO v5', 'COCO', 'VOC'],
        value='YOLO v5',
        description='Format:',
        layout=widgets.Layout(width='40%')
    )
    
    output_dir = widgets.Text(
        value=config.get('data', {}).get('dir', 'data'),
        description='Output Dir:',
        layout=widgets.Layout(width='60%')
    )
    
    output_container = widgets.VBox(
        [output_label, output_format, output_dir],
        layout=widgets.Layout(margin='10px 0', padding='10px', border=f'1px solid {COLORS["border"]}', border_radius='5px')
    )
    
    # Action buttons
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download',
        tooltip="Mulai download dataset",
        layout=widgets.Layout(margin='5px')
    )
    
    check_button = widgets.Button(
        description='Cek Status',
        button_style='info',
        icon='info',
        tooltip="Cek status dan validasi dataset",
        layout=widgets.Layout(margin='5px')
    )
    
    button_container = widgets.HBox(
        [download_button, check_button],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            justify_content='flex-start',
            margin='10px 0'
        )
    )
    
    # PERBAIKAN: Status panel yang terlihat jelas di bawah tombol
    from smartcash.ui.utils.constants import ALERT_STYLES
    status_panel = widgets.HTML(
        value=f"""
        <div style="padding:10px; background-color:{ALERT_STYLES['info']['bg_color']}; 
                   color:{ALERT_STYLES['info']['text_color']}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {ALERT_STYLES['info']['text_color']};">
           <p style="margin:5px 0">{ALERT_STYLES['info']['icon']} Siap untuk memulai download dataset</p>
        </div>
        """,
        layout=widgets.Layout(
            width='100%',
            margin='10px 0'
        )
    )
    
    # PERBAIKAN: Area untuk konfirmasi dialog
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            margin='10px 0'
        )
    )
    
    # Progress tracking
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Proses:',
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            visibility='hidden'  # Hidden by default
        ),
        style={'description_width': 'initial', 'bar_color': COLORS['primary']}
    )
    
    progress_message = widgets.HTML(
        value="",
        layout=widgets.Layout(
            margin='5px 0',
            visibility='hidden'  # Hidden by default
        )
    )
    
    progress_container = widgets.VBox(
        [progress_bar, progress_message],
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Status output area
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["border"]}',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Info box
    info_box = get_dataset_info()
    
    # Container utama dengan semua komponen
    main = widgets.VBox(
        [
            header,
            endpoint_container,
            rf_accordion,
            drive_accordion,
            url_accordion,
            output_container,
            button_container,
            status_panel,              # PERBAIKAN: Status panel di bawah button
            confirmation_area,         # PERBAIKAN: Area konfirmasi sebelum log output
            progress_container,
            status,
            info_box
        ],
        layout=widgets.Layout(
            width='100%',
            padding='10px'
        )
    )
    
    # Struktur final komponen UI
    ui_components = {
        'ui': main,
        'endpoint_dropdown': endpoint_dropdown,
        'rf_workspace': rf_workspace,
        'rf_project': rf_project,
        'rf_version': rf_version,
        'rf_apikey': rf_apikey,
        'drive_folder': drive_folder,
        'url_input': url_input,
        'output_format': output_format,
        'output_dir': output_dir,
        'download_button': download_button,
        'check_button': check_button,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'status': status,
        'rf_accordion': rf_accordion,
        'drive_accordion': drive_accordion,
        'url_accordion': url_accordion,
        'status_panel': status_panel,       # PERBAIKAN: Tambahkan panel status ke UI components
        'confirmation_area': confirmation_area,  # PERBAIKAN: Tambahkan area konfirmasi ke UI components
        'module_name': 'dataset_downloader'
    }
    
    return ui_components
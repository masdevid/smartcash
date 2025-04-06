"""
File: smartcash/ui/dataset/download/download_component.py
Deskripsi: Komponen UI untuk download dataset dengan progres bar dan integrasi observer
"""

import ipywidgets as widgets
import os
from typing import Dict, Any, Optional

def create_dataset_download_ui(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buat komponen UI untuk download dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import MAIN_CONTAINER, OUTPUT_WIDGET, BUTTON
    from smartcash.ui.info_boxes.dataset_info import get_dataset_info
    
    # Header
    header = create_header(
        "📥 Dataset Downloader", 
        "Download dataset untuk training dan validation"
    )
    
    # Endpoint selection
    endpoint_label = widgets.HTML(f"<b>{ICONS['data']} Pilih Endpoint Dataset</b>")
    endpoint_dropdown = widgets.Dropdown(
        options=['Roboflow', 'Google Drive'],
        value='Roboflow',
        description='Sumber:',
        layout=widgets.Layout(width='40%')
    )
    
    endpoint_container = widgets.VBox([endpoint_label, endpoint_dropdown], layout=widgets.Layout(margin='10px 0'))
    
    # Ambil nilai default dari konfigurasi
    roboflow_config = config.get('data', {}).get('roboflow', {})
    rf_workspace_default = roboflow_config.get('workspace', 'smartcash-wo2us')
    rf_project_default = roboflow_config.get('project', 'rupiah-emisi-2022')
    rf_version_default = roboflow_config.get('version', '3')
    api_key_env = os.environ.get('ROBOFLOW_API_KEY', '')
    
    # Roboflow Config
    rf_workspace = widgets.Text(value=rf_workspace_default, placeholder='Workspace ID', description='Workspace:', layout=widgets.Layout(width='80%'))
    rf_project = widgets.Text(value=rf_project_default, placeholder='Project ID', description='Project:', layout=widgets.Layout(width='80%'))
    rf_version = widgets.Text(value=rf_version_default, placeholder='Version', description='Version:', layout=widgets.Layout(width='40%'))
    rf_apikey = widgets.Password(value=api_key_env, placeholder='API Key', description='API Key:', layout=widgets.Layout(width='80%'))
    rf_container = widgets.VBox([rf_workspace, rf_project, rf_version, rf_apikey], layout=widgets.Layout(margin='5px 0'))
    
    rf_accordion = widgets.Accordion([rf_container], selected_index=0, layout=widgets.Layout(width='100%', margin='10px 0'))
    rf_accordion.set_title(0, f"{ICONS['config']} Konfigurasi Roboflow")
    
    # Google Drive Config
    drive_folder = widgets.Text(
        value=config.get('data', {}).get('drive_folder', 'SmartCash/datasets'),
        placeholder='Path folder di Drive',
        description='Folder:',
        layout=widgets.Layout(width='80%')
    )
    drive_container = widgets.VBox([drive_folder], layout=widgets.Layout(margin='5px 0'))
    drive_accordion = widgets.Accordion([drive_container], selected_index=None, layout=widgets.Layout(width='100%', margin='10px 0'))
    drive_accordion.set_title(0, f"{ICONS['folder']} Konfigurasi Google Drive")
    
    # Output lokasi - hanya YOLO v5 format sebagai default
    output_label = widgets.HTML(f"<b>{ICONS['folder']} Output Dataset</b>")
    output_dir = widgets.Text(
        value=config.get('data', {}).get('dir', 'data'),
        description='Output Dir:',
        layout=widgets.Layout(width='60%')
    )
    output_container = widgets.VBox(
        [output_label, output_dir],
        layout=widgets.Layout(margin='10px 0', padding='10px', border=f'1px solid {COLORS["border"]}', border_radius='5px')
    )
    
    # Action buttons
    download_button = widgets.Button(description='Download Dataset', button_style='primary', icon='download', tooltip="Mulai download dataset", layout=BUTTON)
    check_button = widgets.Button(description='Cek Status', button_style='info', icon='info', tooltip="Cek status dan validasi dataset", layout=BUTTON)
    button_container = widgets.HBox([download_button, check_button], layout=widgets.Layout(display='flex', flex_flow='row', justify_content='flex-start', margin='10px 0'))
    
    # Status panel
    from smartcash.ui.utils.constants import ALERT_STYLES
    status_panel = widgets.HTML(
        value=f"""
        <div style="padding:10px; background-color:{ALERT_STYLES['info']['bg_color']}; 
                   color:{ALERT_STYLES['info']['text_color']}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {ALERT_STYLES['info']['text_color']};">
           <p style="margin:5px 0">{ALERT_STYLES['info']['icon']} Siap untuk memulai download dataset</p>
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Area untuk konfirmasi dialog
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # PENTING: Progress tracking yang benar-benar ada
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, description='Proses:',
        layout=widgets.Layout(width='100%', margin='10px 0', visibility='hidden'),
        style={'description_width': 'initial', 'bar_color': COLORS['primary']}
    )
    progress_message = widgets.HTML(value="", layout=widgets.Layout(margin='5px 0', visibility='hidden'))
    progress_container = widgets.VBox([progress_bar, progress_message], layout=widgets.Layout(margin='10px 0'))
    
    # Status output area
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Info box
    info_box = get_dataset_info()
    
    # Container utama dengan semua komponen
    main = widgets.VBox(
        [
            header, endpoint_container, rf_accordion, drive_accordion,
            output_container, button_container, status_panel, confirmation_area,
            progress_container, status, info_box
        ],
        layout=MAIN_CONTAINER
    )
    
    # Struktur final komponen UI dengan format output tetap YOLO v5
    ui_components = {
        'ui': main,
        'endpoint_dropdown': endpoint_dropdown,
        'rf_workspace': rf_workspace,
        'rf_project': rf_project,
        'rf_version': rf_version,
        'rf_apikey': rf_apikey,
        'drive_folder': drive_folder,
        'output_dir': output_dir,
        'download_button': download_button,
        'check_button': check_button,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'status': status,
        'rf_accordion': rf_accordion,
        'drive_accordion': drive_accordion,
        'status_panel': status_panel,
        'confirmation_area': confirmation_area,
        'module_name': 'dataset_download',
        'format': 'yolov5pytorch'  # Default format tetap
    }
    
    return ui_components
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
    # Import komponen UI yang terstandarisasi
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.download_info import get_download_info
    from smartcash.ui.utils.layout_utils import MAIN_CONTAINER, OUTPUT_WIDGET, HORIZONTAL_GROUP
    
    # Header menggunakan komponen standar
    header = create_header(
        f"{ICONS['dataset']} Download Dataset",
        "Download dan persiapan dataset untuk SmartCash"
    )
    
    # Panel info status menggunakan fungsi standar dari utils
    from smartcash.ui.utils.alert_utils import create_info_alert
    status_panel = widgets.HTML(
        value=create_info_alert(
            f"Pilih sumber dataset dan konfigurasi download",
            "info"
        ).value
    )
    
    # Download options lebih ringkas dengan style standar
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
    
    # Tombol aksi menggunakan helper standar
    from smartcash.ui.helpers.ui_helpers import create_button_group
    download_button = widgets.Button(
        description='Download Dataset',
        button_style='primary',
        icon='download'
    )
    
    # Progress container dengan layout standar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Download: 0%',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='75%', visibility='visible')
    )
    
    progress_label = widgets.Label('Siap untuk download dataset')
    progress_container = widgets.HBox(
        [progress_bar, progress_label], 
        layout=HORIZONTAL_GROUP
    )
    
    # Status output dengan layout standar
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Panel info bantuan dengan komponen standar
    help_panel = get_download_info()
    
    # Container untuk download settings yang menggunakan styling konsisten
    download_settings_container = widgets.VBox([roboflow_settings])
    
    # Tambahkan separator standar
    from smartcash.ui.helpers.ui_helpers import create_divider
    separator = create_divider()
    
    # Rakit komponen UI dengan layout konsisten
    ui = widgets.VBox(
        [
            header,
            status_panel,
            download_options,
            download_settings_container,
            separator,
            download_button,
            progress_container,
            status,
            help_panel
        ],
        layout=MAIN_CONTAINER
    )
    
    # Komponen UI dengan penamaan konsisten
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
        'progress_container': progress_container,
        'status': status,
        'status_output': status,  # Alias untuk kompatibilitas
        'help_panel': help_panel,
        'module_name': 'dataset_download'
    }
    
    return ui_components
"""
File: smartcash/ui/dataset/download/components/download_component.py
Deskripsi: Komponen UI utama untuk download dataset menggunakan shared components
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_download_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk download dataset dari Roboflow.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.utils.layout_utils import create_divider
    
    # Import shared components
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    import os
    
    # Ambil konfigurasi jika tersedia
    config = config or {}
    roboflow_config = config.get('data', {}).get('roboflow', {})
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS.get('download', '📥')} Dataset Download", 
                          "Download dataset untuk training model SmartCash")
    
    # Panel info status
    status_panel = create_status_panel("Konfigurasi download dataset", "info")
    
    # Ambil nilai default dari konfigurasi
    rf_workspace_default = roboflow_config.get('workspace', 'smartcash-wo2us')
    rf_project_default = roboflow_config.get('project', 'rupiah-emisi-2022')
    rf_version_default = roboflow_config.get('version', '3')
    api_key_env = os.environ.get('ROBOFLOW_API_KEY', '')
    output_dir_default = config.get('data', {}).get('dir', 'data')
    
    # Roboflow Config
    rf_workspace = widgets.Text(
        value=rf_workspace_default, 
        placeholder='Workspace ID', 
        description='Workspace:', 
        layout=widgets.Layout(width='100%')
    )
    
    rf_project = widgets.Text(
        value=rf_project_default, 
        placeholder='Project ID', 
        description='Project:', 
        layout=widgets.Layout(width='100%')
    )
    
    rf_version = widgets.Text(
        value=rf_version_default, 
        placeholder='Version', 
        description='Version:', 
        layout=widgets.Layout(width='100%')
    )
    
    rf_apikey = widgets.Password(
        value=api_key_env, 
        placeholder='API Key', 
        description='API Key:', 
        layout=widgets.Layout(width='100%')
    )
    
    # Buat input untuk path penyimpanan
    output_dir = widgets.Text(
        value=output_dir_default,
        placeholder='Path penyimpanan dataset',
        description='Output Dir:',
        disabled=False,
        layout=widgets.Layout(width='100%')
    )
    
    # Buat checkbox untuk validasi dataset
    validate_dataset = widgets.Checkbox(
        value=True,
        description='Validasi dataset setelah download',
        disabled=False,
        indent=False,
        layout=widgets.Layout(width='100%')
    )
    
    # Buat container untuk input options dengan padding dan margin yang konsisten
    input_options = widgets.VBox([
        rf_workspace,
        rf_project,
        rf_version,
        rf_apikey,
        output_dir,
        validate_dataset
    ], layout=widgets.Layout(
        width='100%', 
        margin='10px 0', 
        padding='15px', 
        border=f'1px solid {COLORS.get("border", "#ddd")}', 
        border_radius='5px'
    ))
    
    # Buat tombol-tombol download dengan shared component (tanpa redundansi)
    # Catatan: reset_button sudah tersedia secara default dalam create_action_buttons
    action_buttons = create_action_buttons(
        primary_label="Download Dataset",
        primary_icon="download",
        secondary_buttons=[
            ("Check Dataset", "search", "info")
        ],
        cleanup_enabled=True  # Aktifkan tombol cleanup untuk menghapus hasil download jika diperlukan
    )
    
    # Progress tracking dengan shared component dan layout yang konsisten
    # Catatan: Layout akan diatur setelah komponen dibuat
    progress_components = create_progress_tracking(
        module_name='download',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )
    
    # Atur layout untuk progress container setelah dibuat
    progress_components['progress_container'].layout.margin = '15px 0'
    progress_components['progress_container'].layout.padding = '5px 0'
    progress_components['progress_container'].layout.border_radius = '5px'
    
    # Log accordion dengan shared component dan layout yang konsisten
    # Catatan: Layout akan diatur setelah komponen dibuat
    log_components = create_log_accordion(
        module_name='download',
        height='200px',
        width='100%'
    )
    
    # Atur layout untuk log accordion setelah dibuat
    log_components['log_accordion'].layout.margin = '15px 0'
    log_components['log_accordion'].layout.border_radius = '5px'
    
    # Summary stats container dengan styling yang konsisten
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border=f'1px solid {COLORS.get("border", "#ddd")}', 
            padding='15px', 
            margin='15px 0', 
            display='none',
            border_radius='5px',
            min_height='50px'
        )
    )
    
    # Area untuk konfirmasi dialog dengan layout yang konsisten
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            margin='15px 0',
            width='100%',
            min_height='50px',
            padding='5px 0'
        )
    )
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('settings', '⚙️')} Pengaturan Download</h4>"),
        input_options,
        create_divider(),
        action_buttons['container'],
        confirmation_area,  # Tambahkan area konfirmasi sebelum progress
        progress_components['progress_container'],
        log_components['log_accordion'],
        summary_container
    ])
    
    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'rf_workspace': rf_workspace,
        'rf_project': rf_project,
        'rf_version': rf_version,
        'rf_apikey': rf_apikey,
        'output_dir': output_dir,
        'validate_dataset': validate_dataset,
        'input_options': input_options,
        'download_button': action_buttons['primary_button'],
        'check_button': action_buttons['secondary_buttons'][0] if 'secondary_buttons' in action_buttons else None,
        'reset_button': action_buttons.get('reset_button'),  # Menggunakan reset_button dari shared component
        'cleanup_button': action_buttons.get('cleanup_button'),
        'button_container': action_buttons['container'],
        'summary_container': summary_container,
        'confirmation_area': confirmation_area,  # Area konfirmasi untuk dialog
        'module_name': 'download',
        'endpoint_dropdown': {'value': 'Roboflow'}  # Dummy endpoint_dropdown untuk kompatibilitas
    }
    
    # Tambahkan komponen progress tracking
    ui_components.update({
        'progress_bar': progress_components['progress_bar'],
        'progress_container': progress_components['progress_container'],
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label')
    })
    
    # Tambahkan komponen log
    ui_components.update({
        'status': log_components['log_output'],
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion']
    })
    
    return ui_components

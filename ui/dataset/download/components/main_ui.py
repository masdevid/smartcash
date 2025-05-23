"""
File: smartcash/ui/dataset/download/components/main_ui.py  
Deskripsi: Simplified main UI creation yang mempertahankan struktur yang sudah bekerja
"""

import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.common.environment import get_environment_manager
from .options_panel import create_options_panel
from .action_section import create_action_section  # Import yang diperbaiki
from .progress_section import create_progress_section
from .log_section import create_log_section

def create_download_ui(config=None):
    """Create download UI dengan proper component mapping tanpa mengubah struktur yang sudah bekerja."""
    config = config or {}
    roboflow_config = config.get('roboflow', {})
    
    # Environment info untuk Drive status
    env_manager = get_environment_manager()
    drive_status = "🔗 Drive terhubung" if env_manager.is_drive_mounted else "⚠️ Drive tidak terhubung"
    storage_info = f" | Storage: {'Drive' if env_manager.is_drive_mounted else 'Local'}"
    
    # Header dengan storage info
    header = create_header(
        f"{ICONS.get('download', '📥')} Dataset Download", 
        f"Download dataset untuk SmartCash{storage_info}"
    )
    
    # Status panel dengan Drive info
    initial_status = f"{drive_status} - Siap untuk download dataset"
    status_panel = create_status_panel(initial_status, "info")
    
    # Components (mempertahankan struktur yang sudah ada)
    options = create_options_panel(roboflow_config, env_manager)
    actions = create_action_section()  # Menggunakan yang sudah diperbaiki
    progress = create_progress_section()
    logs = create_log_section()
    
    # Main container
    main_container = widgets.VBox([
        header,
        status_panel,
        _create_storage_info_widget(env_manager),
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px;'>{ICONS.get('settings', '⚙️')} Pengaturan Download</h4>"),
        options['panel'],
        widgets.VBox([
            actions['save_reset_buttons']['container'],
            actions['sync_info']['container']
        ], layout=widgets.Layout(align_items='flex-end', width='100%')),
        create_divider(),
        actions['action_buttons']['container'],
        logs['confirmation_area'], 
        progress['progress_container'],
        logs['log_accordion'],
        logs['summary_container']
    ], layout=widgets.Layout(
        width='100%', display='flex', flex_flow='column',
        align_items='stretch', padding='10px', border='1px solid #ddd',
        border_radius='5px', background_color='#fff'
    ))
    
    # 📋 Compose UI components dengan key mapping yang tepat
    ui_components = {
        'ui': main_container,
        'main_container': main_container,
        'header': header,
        'status_panel': status_panel,
        'drive_info': main_container.children[2],  # Storage info widget
        'module_name': 'download',
        'env_manager': env_manager
    }
    
    # 🔗 Add options panel components (unpack semua kecuali 'panel')
    ui_components.update({k: v for k, v in options.items() if k != 'panel'})
    
    # 🔗 Add action components dengan key yang konsisten
    # Ini adalah bagian yang penting - pastikan key sesuai dengan yang diharapkan handler
    ui_components.update({
        'download_button': actions['download_button'],    # Handler expects this key
        'check_button': actions['check_button'],          # Handler expects this key  
        'reset_button': actions['reset_button'],          # Handler expects this key
        'save_button': actions['save_button'],            # Handler expects this key
        'cleanup_button': actions.get('cleanup_button'), # Handler expects this key (optional)
    })
    
    # 🔗 Add progress components
    ui_components.update({k: v for k, v in progress.items()})
    
    # 🔗 Add log components
    ui_components.update({k: v for k, v in logs.items()})
    
    # 🔗 Add action section references (untuk debugging)
    ui_components.update({
        'actions': actions,
        'save_reset_buttons': actions['save_reset_buttons'],
        'action_buttons': actions['action_buttons']
    })
    
    return ui_components

def _create_storage_info_widget(env_manager):
    """Create widget untuk info storage (tidak berubah)."""
    if env_manager.is_drive_mounted:
        info_html = f"""
        <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #2e7d32;">✅ Dataset akan disimpan di Google Drive: {env_manager.drive_path}</span>
        </div>
        """
    else:
        info_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #856404;">⚠️ Drive tidak terhubung - dataset akan disimpan lokal (hilang saat restart)</span>
        </div>
        """
    return widgets.HTML(info_html)
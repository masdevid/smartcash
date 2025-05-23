"""
File: smartcash/ui/dataset/download/components/main_ui.py
Deskripsi: Updated main UI dengan Drive storage info dan progress observer
"""

import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.common.environment import get_environment_manager
from .options_panel import create_options_panel
from .action_section import create_action_section
from .progress_section import create_progress_section
from .log_section import create_log_section

def create_download_ui(config=None):
    """Create download UI dengan Drive storage integration."""
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
    
    # Components
    options = create_options_panel(roboflow_config, env_manager)
    actions = create_action_section()
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
    
    # Compose UI components
    return {
        'ui': main_container,
        'main_container': main_container,
        'header': header,
        'status_panel': status_panel,
        'drive_info': main_container.children[2],  # Storage info widget
        **{k: v for k, v in options.items() if k != 'panel'},
        **{k: v for k, v in actions.items()},
        **{k: v for k, v in progress.items()},
        **{k: v for k, v in logs.items()},
        'module_name': 'download',
        'env_manager': env_manager
    }

def _create_storage_info_widget(env_manager):
    """Create widget untuk info storage."""
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
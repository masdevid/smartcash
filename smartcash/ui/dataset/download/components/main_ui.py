"""
File: smartcash/ui/dataset/download/components/main_ui.py
Deskripsi: Fixed main UI creation dengan proper component mapping dan consistent naming
"""

import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.common.environment import get_environment_manager
from .options_panel import create_options_panel
from .progress_section import create_progress_section
from .log_section import create_log_section

def create_download_ui(config=None):
    """Create download UI dengan proper component mapping dan consistent naming."""
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
    progress = create_progress_section()
    logs = create_log_section()
    
    # 🔘 Create action buttons dengan proper naming
    action_buttons = _create_action_buttons_section()
    save_reset_buttons = _create_save_reset_section()
    
    # Main container
    main_container = widgets.VBox([
        header,
        status_panel,
        _create_storage_info_widget(env_manager),
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px;'>{ICONS.get('settings', '⚙️')} Pengaturan Download</h4>"),
        options['panel'],
        save_reset_buttons['container'],
        create_divider(),
        action_buttons['container'],
        logs['confirmation_area'],
        progress['progress_container'],
        logs['log_accordion'],
        logs['summary_container']
    ], layout=widgets.Layout(
        width='100%', display='flex', flex_flow='column',
        align_items='stretch', padding='10px', border='1px solid #ddd',
        border_radius='5px', background_color='#fff'
    ))
    
    # 📋 Compose UI components dengan proper key mapping
    ui_components = {
        'ui': main_container,
        'main_container': main_container,
        'header': header,
        'status_panel': status_panel,
        'drive_info': main_container.children[2],  # Storage info widget
        'module_name': 'download',
        'env_manager': env_manager
    }
    
    # 🔗 Add options panel components
    ui_components.update({k: v for k, v in options.items() if k != 'panel'})
    
    # 🔗 Add action button components (dengan key yang konsisten)
    ui_components.update({
        'download_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'], 
        'cleanup_button': action_buttons.get('cleanup_button'),  # Might not exist
    })
    
    # 🔗 Add save/reset button components
    ui_components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button']
    })
    
    # 🔗 Add progress components
    ui_components.update({k: v for k, v in progress.items()})
    
    # 🔗 Add log components
    ui_components.update({k: v for k, v in logs.items()})
    
    return ui_components

def _create_action_buttons_section():
    """Create action buttons section dengan proper component creation."""
    from smartcash.ui.components.action_buttons import create_action_buttons
    
    return create_action_buttons(
        primary_label="Download Dataset",
        primary_icon="download",
        secondary_buttons=[
            ("Check Dataset", "search", "info")
        ],
        cleanup_enabled=True,
        button_width='140px'
    )

def _create_save_reset_section():
    """Create save/reset buttons section."""
    from smartcash.ui.components.action_buttons import create_save_reset_action_buttons
    
    return create_save_reset_action_buttons(
        save_label='Simpan',
        reset_label='Reset',
        button_width='100px'
    )

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
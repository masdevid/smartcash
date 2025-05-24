"""
File: smartcash/ui/dataset/download/components/main_ui.py  
Deskripsi: Fixed progress container dengan visibility controls yang proper
"""

import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.common.environment import get_environment_manager

from .options_panel import create_options_panel
from .action_section import create_action_section
from .log_section import create_log_section

def create_download_ui(config=None):
    """Create download UI dengan progress container dan controls yang bekerja."""
    config = config or {}
    roboflow_config = config.get('roboflow', {})
    
    # Environment info
    env_manager = get_environment_manager()
    drive_status = "🔗 Drive terhubung" if env_manager.is_drive_mounted else "⚠️ Drive tidak terhubung"
    storage_info = f" | Storage: {'Drive' if env_manager.is_drive_mounted else 'Local'}"
    
    # Components
    header = create_header(
        f"{ICONS.get('download', '📥')} Dataset Download", 
        f"Download dataset untuk SmartCash{storage_info}"
    )
    
    initial_status = f"{drive_status} - Siap untuk download dataset"
    status_panel = create_status_panel(initial_status, "info")
    
    options = create_options_panel(roboflow_config, env_manager)
    actions = create_action_section()
    logs = create_log_section()
    
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi download saat ini",
        reset_tooltip="Reset konfigurasi ke pengaturan default",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan saat disimpan atau direset.",
        button_width="120px",
        container_width="100%"
    )
    
    # Progress container dengan controls
    progress_components = _create_progress_with_controls()
    
    # Main container
    main_container = widgets.VBox([
        header,
        status_panel,
        _create_storage_info_widget(env_manager),
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px;'>{ICONS.get('settings', '⚙️')} Pengaturan Download</h4>"),
        options['panel'],
        save_reset_buttons['container'],
        create_divider(),
        actions['action_buttons']['container'],
        logs['confirmation_area'], 
        progress_components['container'],
        logs['log_accordion'],
        logs['summary_container']
    ], layout=widgets.Layout(
        width='100%', display='flex', flex_flow='column',
        align_items='stretch', padding='10px', border='1px solid #ddd',
        border_radius='5px', background_color='#fff'
    ))
    
    # UI components dengan progress controls
    ui_components = {
        'ui': main_container,
        'main_container': main_container,
        'header': header,
        'status_panel': status_panel,
        'drive_info': main_container.children[2],
        'module_name': 'download',
        'env_manager': env_manager,
    }
    
    # Add progress components dan controls
    ui_components.update(progress_components)
    
    # Add options, actions, save/reset, logs
    ui_components.update({k: v for k, v in options.items() if k != 'panel'})
    ui_components.update({
        'download_button': actions['download_button'],
        'check_button': actions['check_button'],
        'cleanup_button': actions.get('cleanup_button'),
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
    })
    ui_components.update({k: v for k, v in logs.items()})
    
    return ui_components

def _create_progress_with_controls():
    """Create progress container dengan visibility controls yang dibutuhkan handlers."""
    
    # Progress widgets
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Overall:',
        bar_style='info',
        layout=widgets.Layout(width='100%', height='20px', visibility='hidden')
    )
    
    overall_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='2px 0', visibility='hidden')
    )
    
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Step:',
        bar_style='info',
        layout=widgets.Layout(width='100%', height='20px', visibility='hidden')
    )
    
    step_label = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='2px 0', visibility='hidden')
    )
    
    # Container
    container = widgets.VBox([
        widgets.HTML("<h4>📊 Progress</h4>"),
        progress_bar, 
        overall_label,
        current_progress, 
        step_label
    ], layout=widgets.Layout(
        margin='10px 0', 
        padding='10px', 
        display='none',  # Hidden by default
        border='1px solid #ddd',
        border_radius='5px'
    ))
    
    # Control functions
    def show_container():
        container.layout.display = 'block'
        container.layout.visibility = 'visible'
    
    def hide_container():
        container.layout.display = 'none'
        container.layout.visibility = 'hidden'
    
    def show_for_operation(operation):
        """Show progress sesuai operation type."""
        show_container()
        if operation in ['download', 'check', 'cleanup']:
            progress_bar.layout.visibility = 'visible'
            overall_label.layout.visibility = 'visible'
            if operation == 'download':
                current_progress.layout.visibility = 'visible'
                step_label.layout.visibility = 'visible'
    
    def update_progress(progress_type, value, message="", color_style=None):
        """Update progress dengan type dan message."""
        value = max(0, min(100, value))
        
        if progress_type == 'overall':
            progress_bar.value = value
            progress_bar.description = f'Overall: {value}%'
            if color_style:
                progress_bar.bar_style = color_style
            if message:
                overall_label.value = f"<div style='color: #495057; font-size: 13px;'>{message}</div>"
        
        elif progress_type == 'current' or progress_type == 'step':
            current_progress.value = value
            current_progress.description = f'Step: {value}%'
            if color_style:
                current_progress.bar_style = color_style
            if message:
                step_label.value = f"<div style='color: #495057; font-size: 13px;'>{message}</div>"
    
    def complete_operation(message="Selesai"):
        """Complete operation dengan success state."""
        update_progress('overall', 100, f"✅ {message}", 'success')
        update_progress('step', 100, f"✅ {message}", 'success')
    
    def error_operation(message="Error"):
        """Set error state."""
        update_progress('overall', 0, f"❌ {message}", 'danger')
        update_progress('step', 0, f"❌ {message}", 'danger')
    
    def reset_all():
        """Reset semua progress."""
        progress_bar.value = 0
        current_progress.value = 0
        progress_bar.bar_style = 'info'
        current_progress.bar_style = 'info'
        overall_label.value = ""
        step_label.value = ""
        hide_container()
    
    return {
        'container': container,
        'progress_container': container,  # Alias
        'progress_bar': progress_bar,
        'overall_label': overall_label,
        'current_progress': current_progress,
        'step_label': step_label,
        
        # Control functions yang dibutuhkan handlers
        'show_container': show_container,
        'hide_container': hide_container,
        'show_for_operation': show_for_operation,
        'update_progress': update_progress,
        'complete_operation': complete_operation,
        'error_operation': error_operation,
        'reset_all': reset_all
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
"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Fixed UI components dengan confirmation area yang visible
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from .input_options import create_downloader_input_options


def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan visible confirmation area"""
    
    get_icon = lambda key, fallback="📥": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('download', '📥')} Dataset Downloader", 
        "Download dataset Roboflow untuk SmartCash training dengan UUID renaming otomatis"
    )
    
    # Status panel
    status_panel = create_status_panel("Siap untuk download dataset", "info")
    
    # Input options
    input_options = create_downloader_input_options(config)
    
    # Action buttons
    action_buttons = create_action_buttons(
        primary_label="Download Dataset",
        primary_icon="download",
        secondary_buttons=[("Check Dataset", "search", "info")],
        cleanup_enabled=True,
        button_width='130px'
    )
    # Confirmation area - VISIBLE dan di bawah action buttons seperti preprocessing
    confirmation_area = widgets.Output(
        layout=widgets.Layout(
            width='100%', 
            min_height='0px', 
            max_height='800px',
            margin='10px 0',
            padding='8px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            background_color='#fafafa',
            overflow='auto',
            visibility='hidden',  # Start hidden
            display='block'       # Ensure it can be shown
        )
    )
    
    # Save & reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        save_tooltip="Simpan konfigurasi downloader",
        reset_tooltip="Reset konfigurasi ke default",
        with_sync_info=True,
        sync_message="Konfigurasi tersimpan ke dataset_config.yaml"
    )
    
    # Log accordion
    log_components = create_log_accordion(module_name='downloader', height='220px')
    
    # Progress tracking dengan dual level untuk overall dan step progress
    progress_tracker = create_dual_progress_tracker(operation="Dataset Download")
    
    # Help panel
    help_content = """
    <div style="padding: 8px; background: #ffffff;">
        <p style="margin: 6px 0; font-size: 13px;">Download dataset dari Roboflow dengan UUID renaming dan validasi otomatis.</p>
        <div style="margin: 8px 0;">
            <strong style="color: #495057; font-size: 13px;">Parameter Utama:</strong>
            <ul style="margin: 4px 0; padding-left: 18px; color: #495057; font-size: 12px;">
                <li><strong>Workspace/Project:</strong> Identifikasi dataset Roboflow</li>
                <li><strong>Version:</strong> Versi dataset yang akan didownload</li>
                <li><strong>API Key:</strong> Auto-detect dari Colab secrets</li>
                <li><strong>UUID Renaming:</strong> Penamaan ulang otomatis untuk konsistensi</li>
            </ul>
        </div>
        <div style="margin-top: 8px; padding: 6px; background: #e7f3ff; border-radius: 3px; font-size: 12px;">
            <strong>💡 Tips:</strong> API key dapat diatur di Colab Secrets dengan nama 'ROBOFLOW_API_KEY'.
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Info Download")
    help_panel.selected_index = None
    
    # Section headers
    config_header = widgets.HTML(f"""
        <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 8px 0; font-size: 16px;'>
            {get_icon('settings', '⚙️')} Konfigurasi Download
        </h4>
    """)
    
    action_header = widgets.HTML(f"""
    <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 10px 0; font-size: 16px; 
               border-bottom: 2px solid {get_color('primary', '#007bff')}; padding-bottom: 6px;'>
        {get_icon('play', '▶️')} Actions
    </h4>
    """)
    
    # Main UI assembly dengan confirmation area yang visible
    ui = widgets.VBox([
        header, 
        status_panel, 
        config_header, 
        input_options, 
        save_reset_buttons['container'], 
        action_header, 
        action_buttons['container'],
        confirmation_area,  # Visible confirmation area di bawah action buttons
        progress_tracker.container, 
        log_components['log_accordion'], 
        create_divider(), 
        help_panel
    ], layout=widgets.Layout(width='100%', padding='8px', overflow='hidden'))
    
    # Compile components
    ui_components = {
        # Main UI
        'ui': ui, 'header': header, 'status_panel': status_panel,
        
        # Input components
        'input_options': input_options,
        'workspace_input': getattr(input_options, 'workspace_input', None),
        'project_input': getattr(input_options, 'project_input', None), 
        'version_input': getattr(input_options, 'version_input', None),
        'api_key_input': getattr(input_options, 'api_key_input', None),
        'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        
        # Action buttons
        'action_buttons': action_buttons,
        'download_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Confirmation area - IMPORTANT!
        'confirmation_area': confirmation_area,
        
        # Save/reset buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Progress components
        'progress_tracker': progress_tracker,
        'progress_container': progress_tracker.container,
        'show_for_operation': progress_tracker.show,
        'update_progress': progress_tracker.update,
        'complete_operation': progress_tracker.complete,
        'error_operation': progress_tracker.error,
        'reset_all': progress_tracker.reset,
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],
        
        # UI info
        'help_panel': help_panel,
        'module_name': 'downloader'
    }
    
    # Validate critical components - create fallback buttons if missing
    critical_components = ['download_button', 'check_button', 'save_button', 'reset_button']
    for comp_name in critical_components:
        if ui_components.get(comp_name) is None:
            ui_components[comp_name] = widgets.Button(
                description=comp_name.replace('_', ' ').title(),
                button_style='primary' if 'download' in comp_name else '',
                disabled=True,
                tooltip=f"Component {comp_name} tidak tersedia",
                layout=widgets.Layout(width='auto', max_width='150px')
            )
    
    return ui_components
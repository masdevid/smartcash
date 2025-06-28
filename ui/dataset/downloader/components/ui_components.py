"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Fixed UI components dengan confirmation area yang visible
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components import (
    create_action_buttons, create_dual_progress_tracker, create_divider,
    create_status_panel, create_log_accordion, create_save_reset_buttons
)
from smartcash.ui.components.dialog import create_confirmation_area
from .input_options import create_downloader_input_options


def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan visible confirmation area"""
    # Initialize ui_components dictionary to store all UI components
    ui_components = {}
    
    get_icon = lambda key, fallback="📥": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        "Dataset Downloader", 
        "Download dataset Roboflow untuk SmartCash training dengan UUID renaming otomatis",
        "📥"
    )
    
    # Status panel
    status_panel = create_status_panel("Siap untuk download dataset", "info")
    
    # Input options
    input_options = create_downloader_input_options(config)
    
    # Action buttons with new API
    action_buttons = create_action_buttons(
        primary_button={
            "label": "📥 Download Dataset",
            "style": "primary",
            "width": "140px"
        },
        secondary_buttons=[
            {
                "label": "🔍 Check Dataset",
                "style": "info",
                "width": "140px"
            },
            {
                "label": "🗑️ Bersihkan Dataset",
                "style": "warning",
                "tooltip": "Hapus dataset yang sudah didownload",
                "width": "140px"
            }
        ]
    )
    
    # Get buttons from the new action buttons component
    download_button = action_buttons.get('primary')
    check_button = action_buttons.get('secondary_0')
    cleanup_button = action_buttons.get('secondary_1')
    button_container = action_buttons['container']
    
    # Store action buttons in ui_components
    ui_components.update({
        'action_buttons': action_buttons,
        'download_button': download_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'button_container': button_container,
        'buttons': [download_button, check_button, cleanup_button] if cleanup_button else [download_button, check_button]
    })
    
    # Initialize confirmation area using the dialog manager
    # This creates or gets the confirmation area with proper layout and behavior
    confirmation_area = create_confirmation_area(ui_components)
    ui_components['confirmation_area'] = confirmation_area
    
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
    
    # Section headers with consistent styling
    config_header = widgets.HTML(
        "<div style='font-weight:bold;color:#28a745;margin-bottom:8px;'>"
        "⚙️ Konfigurasi Download"
        "</div>"
    )
    
    action_header = widgets.HTML(
        "<div style='font-weight:bold;color:#28a745;margin-bottom:8px;'>"
        "🚀 Actions"
        "</div>"
    )
    
    # === LAYOUT SECTIONS ===
    
    # Config section with save/reset buttons
    config_section = widgets.VBox([
        widgets.Box([save_reset_buttons['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(margin='8px 0'))
    
    # Import shared action section component
    from smartcash.ui.components.action_section import create_action_section
    
    # Create action section using shared component
    action_section = create_action_section(
        action_buttons=ui_components['action_buttons'],
        confirmation_area=confirmation_area,
        title="🚀 Operations",
        status_label="📋 Status & Konfirmasi:",
        show_status=True
    )
    
    # Help section
    help_section = widgets.VBox([
        help_panel
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0 0 0'
    ))
    
    # Main UI assembly with consistent styling
    ui = widgets.VBox([
        # Header section
        header,
        status_panel,
        
        # Config section
        input_options,
        config_section,
        
        # Action section with confirmation area
        action_section,
        
        # Progress tracker
        progress_tracker.container if hasattr(progress_tracker, 'container') else widgets.VBox([]),
        
        # Logs and help sections
        log_components['log_accordion'],
        help_section
    ], layout=widgets.Layout(
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='15px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        box_shadow='0 2px 4px rgba(0,0,0,0.05)'
    ))
    
    # Update ui_components with remaining components
    ui_components.update({
        # Main UI
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'input_options': input_options,
        'config_section': config_section,
        'action_section': action_section,
        'help_section': help_section,
        'help_panel': help_panel,
        'module_name': 'downloader',
        
        # Input components
        'workspace_input': getattr(input_options, 'workspace_input', None),
        'project_input': getattr(input_options, 'project_input', None), 
        'version_input': getattr(input_options, 'version_input', None),
        'api_key_input': getattr(input_options, 'api_key_input', None),
        'validate_checkbox': getattr(input_options, 'validate_checkbox', None),
        'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
        
        # Save/reset buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons.get('save_button'),
        'reset_button': save_reset_buttons.get('reset_button'),
        
        # Progress components
        'progress_tracker': progress_tracker,
        'progress_container': progress_tracker.container if hasattr(progress_tracker, 'container') else None,
        'show_for_operation': getattr(progress_tracker, 'show', None),
        'update_progress': getattr(progress_tracker, 'update', None),
        'complete_operation': getattr(progress_tracker, 'complete', None),
        'error_operation': getattr(progress_tracker, 'error', None),
        'reset_all': getattr(progress_tracker, 'reset', None),
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components.get('log_accordion'),
        'log_output': log_components.get('log_output'),
        'status': log_components.get('log_output')
    })
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    return ui_components
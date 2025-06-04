"""
File: smartcash/ui/dataset/downloader/components/ui_layout.py
Deskripsi: Fixed UI layout dengan proper error handling dan simplified imports
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.fallback_utils import try_operation_safe, create_fallback_ui

def create_downloader_ui(config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Create downloader UI dengan comprehensive error handling."""
    config = config or {}
    
    try:
        # Create basic components dengan error handling
        ui_components = {}
        
        # Header
        ui_components['header'] = try_operation_safe(
            lambda: _create_header(),
            fallback_value=widgets.HTML("<h3>📥 Dataset Downloader</h3>")
        )
        
        # Status panel
        ui_components['status_panel'] = try_operation_safe(
            lambda: _create_status_panel(env),
            fallback_value=widgets.HTML("<div>Status: Ready</div>")
        )
        
        # Form fields
        form_fields = try_operation_safe(
            lambda: _create_form_fields(config),
            fallback_value=_create_basic_form_fields(config)
        )
        ui_components.update(form_fields)
        
        # Action buttons
        action_buttons = try_operation_safe(
            lambda: _create_action_buttons(),
            fallback_value=_create_basic_buttons()
        )
        ui_components.update(action_buttons)
        
        # Log output
        ui_components['log_output'] = widgets.Output(
            layout=widgets.Layout(width='100%', max_height='200px', border='1px solid #ddd')
        )
        
        # Confirmation area
        ui_components['confirmation_area'] = widgets.Output(
            layout=widgets.Layout(width='100%', min_height='50px', display='none')
        )
        
        # Save/Reset buttons
        save_reset = try_operation_safe(
            lambda: _create_save_reset_buttons(),
            fallback_value=_create_basic_save_reset()
        )
        ui_components.update(save_reset)
        
        # Main layout
        ui_components['ui'] = _create_main_layout(ui_components)
        ui_components['main_container'] = ui_components['ui']
        
        return ui_components
        
    except Exception as e:
        # Return fallback UI jika ada error
        return create_fallback_ui(f"Error creating downloader UI: {str(e)}", 'downloader')

def _create_header() -> widgets.HTML:
    """Create header widget."""
    return widgets.HTML("""
    <div style="background: linear-gradient(90deg, #007bff, #0056b3); padding: 15px; color: white; 
               border-radius: 5px; margin-bottom: 15px;">
        <h3 style="margin: 0; color: white;">📥 Dataset Downloader</h3>
        <p style="margin: 5px 0 0; opacity: 0.9;">Download dataset untuk SmartCash training</p>
    </div>
    """)

def _create_status_panel(env=None) -> widgets.HTML:
    """Create status panel dengan environment info."""
    try:
        # Try to get environment info
        if env and hasattr(env, 'is_drive_mounted'):
            storage_info = "Drive terhubung" if env.is_drive_mounted else "Local storage"
        else:
            storage_info = "Storage ready"
        
        status_html = f"""
        <div style="padding: 10px; background: #e3f2fd; border-left: 4px solid #2196f3; 
                   border-radius: 4px; margin-bottom: 15px;">
            <span style="color: #1976d2;">📊 Status: {storage_info} - Siap untuk download</span>
        </div>
        """
        return widgets.HTML(status_html)
    except Exception:
        return widgets.HTML("<div style='padding:10px; background:#f8f9fa;'>📊 Status: Ready</div>")

def _create_form_fields(config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create form fields dengan config defaults."""
    try:
        # Import form creation dengan fallback
        from smartcash.ui.dataset.downloader.components.ui_form import create_form_fields
        return create_form_fields(config)
    except ImportError:
        return _create_basic_form_fields(config)

def _create_basic_form_fields(config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create basic form fields sebagai fallback."""
    roboflow = config.get('roboflow', {})
    local = config.get('local', {})
    
    return {
        'workspace_field': widgets.Text(
            value=roboflow.get('workspace', 'smartcash-wo2us'),
            description='Workspace:',
            layout=widgets.Layout(width='100%')
        ),
        'project_field': widgets.Text(
            value=roboflow.get('project', 'rupiah-emisi-2022'),
            description='Project:',
            layout=widgets.Layout(width='100%')
        ),
        'version_field': widgets.Text(
            value=str(roboflow.get('version', '3')),
            description='Version:',
            layout=widgets.Layout(width='100%')
        ),
        'api_key_field': widgets.Password(
            value=roboflow.get('api_key', ''),
            description='API Key:',
            layout=widgets.Layout(width='100%')
        ),
        'output_dir_field': widgets.Text(
            value=local.get('output_dir', '/content/data'),
            description='Output Dir:',
            layout=widgets.Layout(width='100%')
        ),
        'backup_dir_field': widgets.Text(
            value=local.get('backup_dir', '/content/data/backup'),
            description='Backup Dir:',
            layout=widgets.Layout(width='100%')
        ),
        'organize_dataset': widgets.Checkbox(
            value=local.get('organize_dataset', True),
            description='Organize dataset'
        ),
        'backup_checkbox': widgets.Checkbox(
            value=local.get('backup_enabled', False),
            description='Enable backup'
        )
    }

def _create_action_buttons() -> Dict[str, widgets.Widget]:
    """Create action buttons."""
    download_button = widgets.Button(
        description='Download Dataset',
        icon='download',
        button_style='primary',
        layout=widgets.Layout(width='auto', margin='0 5px 0 0')
    )
    
    check_button = widgets.Button(
        description='Check Dataset',
        icon='search',
        button_style='info',
        layout=widgets.Layout(width='auto', margin='0 5px 0 0')
    )
    
    cleanup_button = widgets.Button(
        description='Hapus Hasil',
        icon='trash',
        button_style='danger',
        layout=widgets.Layout(width='auto')
    )
    
    button_container = widgets.HBox([download_button, check_button, cleanup_button])
    
    return {
        'download_button': download_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'button_container': button_container
    }

def _create_basic_buttons() -> Dict[str, widgets.Widget]:
    """Create basic buttons sebagai fallback."""
    download_button = widgets.Button(description='Download', button_style='primary')
    check_button = widgets.Button(description='Check', button_style='info')
    cleanup_button = widgets.Button(description='Clean', button_style='danger')
    
    return {
        'download_button': download_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'button_container': widgets.HBox([download_button, check_button, cleanup_button])
    }

def _create_save_reset_buttons() -> Dict[str, widgets.Widget]:
    """Create save/reset buttons."""
    try:
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        return create_save_reset_buttons()
    except ImportError:
        return _create_basic_save_reset()

def _create_basic_save_reset() -> Dict[str, widgets.Widget]:
    """Create basic save/reset buttons sebagai fallback."""
    save_button = widgets.Button(description='Simpan', button_style='success')
    reset_button = widgets.Button(description='Reset', button_style='warning')
    
    return {
        'save_button': save_button,
        'reset_button': reset_button,
        'container': widgets.HBox([save_button, reset_button])
    }

def _create_main_layout(ui_components: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create main layout dari components."""
    
    # Form section
    form_section = widgets.VBox([
        widgets.HTML("<h4>📋 Dataset Information</h4>"),
        ui_components.get('workspace_field'),
        ui_components.get('project_field'),
        ui_components.get('version_field'),
        ui_components.get('api_key_field'),
        widgets.HTML("<h4>📁 Storage Settings</h4>"),
        ui_components.get('output_dir_field'),
        ui_components.get('backup_dir_field'),
        widgets.HTML("<h4>⚙️ Options</h4>"),
        ui_components.get('organize_dataset'),
        ui_components.get('backup_checkbox')
    ])
    
    # Log section
    log_accordion = widgets.Accordion([ui_components['log_output']])
    log_accordion.set_title(0, '📋 Log')
    log_accordion.selected_index = None
    
    # Main container
    main_container = widgets.VBox([
        ui_components['header'],
        ui_components['status_panel'],
        form_section,
        ui_components.get('container', widgets.HBox()),  # Save/reset buttons
        ui_components.get('button_container', widgets.HBox()),  # Action buttons
        ui_components['confirmation_area'],
        log_accordion
    ], layout=widgets.Layout(
        width='100%',
        padding='15px',
        border='1px solid #ddd',
        border_radius='5px'
    ))
    
    return main_container
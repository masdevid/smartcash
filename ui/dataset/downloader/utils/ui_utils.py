"""
File: smartcash/ui/dataset/downloader/utils/ui_utils.py
Deskripsi: UI-related utilities untuk downloader dengan widget helpers dan status updates
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

def update_download_button_state(ui_components: Dict[str, Any], state: str = 'ready') -> None:
    """
    Update download button state dengan visual feedback.
    
    Args:
        ui_components: UI components dictionary
        state: Button state ('ready', 'downloading', 'completed', 'error')
    """
    download_button = ui_components.get('download_button')
    if not download_button:
        return
    
    # Button state configurations
    button_states = {
        'ready': {'description': '📥 Download', 'button_style': 'primary', 'disabled': False},
        'downloading': {'description': '⏳ Downloading...', 'button_style': 'warning', 'disabled': True},
        'completed': {'description': '✅ Completed', 'button_style': 'success', 'disabled': False},
        'error': {'description': '❌ Error', 'button_style': 'danger', 'disabled': False}
    }
    
    # Apply state dengan one-liner
    config = button_states.get(state, button_states['ready'])
    [setattr(download_button, attr, value) for attr, value in config.items()]

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """Update status panel dengan message dan styling"""
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        return
    
    # Status styling dengan one-liner
    status_styles = {
        'info': {'color': '#1976d2', 'bg': '#e3f2fd', 'icon': 'ℹ️'},
        'success': {'color': '#2e7d32', 'bg': '#e8f5e8', 'icon': '✅'},
        'warning': {'color': '#856404', 'bg': '#fff3cd', 'icon': '⚠️'},
        'error': {'color': '#721c24', 'bg': '#f8d7da', 'icon': '❌'}
    }
    
    style = status_styles.get(status_type, status_styles['info'])
    
    status_panel.value = f"""
    <div style="padding: 12px; background: {style['bg']}; border-left: 4px solid {style['color']}; 
                border-radius: 4px; margin-bottom: 15px;">
        <span style="color: {style['color']};">{style['icon']} {message}</span>
    </div>
    """

def show_progress_container(ui_components: Dict[str, Any], show: bool = True) -> None:
    """Show/hide progress container dengan one-liner display toggle"""
    progress_container = ui_components.get('progress_container')
    if progress_container:
        progress_container.layout.display = 'block' if show else 'none'

def disable_action_buttons(ui_components: Dict[str, Any], disabled: bool = True) -> None:
    """Disable/enable action buttons dengan one-liner batch update"""
    action_buttons = ['download_button', 'check_button', 'cleanup_button']
    [setattr(ui_components[btn], 'disabled', disabled) for btn in action_buttons 
     if btn in ui_components and hasattr(ui_components[btn], 'disabled')]

def reset_form_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset form fields ke default values dengan safe operations"""
    from smartcash.ui.dataset.downloader.handlers.defaults import get_roboflow_defaults, get_download_defaults
    
    # Get defaults
    roboflow_defaults = get_roboflow_defaults()
    download_defaults = get_download_defaults()
    
    # Reset form fields dengan one-liner safe updates
    field_updates = [
        ('workspace_input', roboflow_defaults['workspace']),
        ('project_input', roboflow_defaults['project']),
        ('version_input', roboflow_defaults['version']),
        ('api_key_input', roboflow_defaults['api_key']),
        ('validate_checkbox', download_defaults['validate_download']),
        ('backup_checkbox', download_defaults['backup_existing'])
    ]
    
    [_safe_set_widget_value(ui_components, widget_name, value) for widget_name, value in field_updates]

def _safe_set_widget_value(ui_components: Dict[str, Any], widget_name: str, value: Any) -> None:
    """Safe widget value setter dengan error handling"""
    widget = ui_components.get(widget_name)
    if widget and hasattr(widget, 'value'):
        try:
            widget.value = value
        except Exception:
            pass  # Silent fail untuk widget update issues

def create_confirmation_message(title: str, details: Dict[str, Any]) -> str:
    """Create formatted confirmation message dengan details"""
    message_parts = [f"<h4>{title}</h4>"]
    
    for key, value in details.items():
        if isinstance(value, dict):
            message_parts.append(f"<h5>{key.title()}:</h5>")
            [message_parts.append(f"  • {k}: {v}") for k, v in value.items()]
        else:
            message_parts.append(f"<strong>{key.title()}:</strong> {value}")
    
    return "<br>".join(message_parts)

def get_form_validation_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get form validation summary untuk UI feedback"""
    from smartcash.ui.dataset.downloader.handlers.validation_handler import validate_complete_form
    return validate_complete_form(ui_components)

def update_api_key_status_display(ui_components: Dict[str, Any], api_key_status: Dict[str, str]) -> None:
    """Update API key status display dengan visual feedback"""
    # This could update a status indicator near the API key field
    # Implementation depends on UI design
    pass

def create_download_summary_message(config: Dict[str, Any]) -> str:
    """Create download summary message untuk confirmation dialog"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    
    return f"""
    <h4>📥 Download Configuration</h4>
    <p><strong>Workspace:</strong> {roboflow.get('workspace', '')}</p>
    <p><strong>Project:</strong> {roboflow.get('project', '')}</p>
    <p><strong>Version:</strong> {roboflow.get('version', '')}</p>
    <p><strong>API Key:</strong> {'✅ Configured' if roboflow.get('api_key') else '❌ Missing'}</p>
    <hr>
    <p><strong>Options:</strong></p>
    <p>• UUID Renaming: {'✅ Enabled' if download.get('rename_files', True) else '❌ Disabled'}</p>
    <p>• Validation: {'✅ Enabled' if download.get('validate_download', True) else '❌ Disabled'}</p>
    <p>• Backup Existing: {'✅ Yes' if download.get('backup_existing', False) else '❌ No'}</p>
    """

def create_cleanup_summary_message(cleanup_info: Dict[str, Any]) -> str:
    """Create cleanup summary message untuk confirmation dialog"""
    total_files = cleanup_info.get('total_files', 0)
    total_size = cleanup_info.get('total_size_mb', 0)
    
    message_parts = [
        f"<h4>🧹 Cleanup Summary</h4>",
        f"<p><strong>Total Files:</strong> {total_files}</p>",
        f"<p><strong>Total Size:</strong> {total_size:.1f}MB</p>",
        "<hr>",
        "<p><strong>Directories to clean:</strong></p>"
    ]
    
    # Add directory details
    for dir_name, dir_info in cleanup_info.get('directories', {}).items():
        if dir_info.get('file_count', 0) > 0:
            message_parts.append(f"• {dir_name.title()}: {dir_info['file_count']} files ({dir_info['size_mb']:.1f}MB)")
    
    return "<br>".join(message_parts)

def toggle_advanced_options(ui_components: Dict[str, Any], show: bool = True) -> None:
    """Toggle advanced options visibility (if implemented in UI)"""
    # Placeholder untuk advanced options toggle
    # Implementation depends on UI design
    pass

def update_form_field_status(ui_components: Dict[str, Any], field_name: str, 
                           status: str = 'normal', message: str = '') -> None:
    """Update individual form field status dengan visual feedback"""
    # This could add validation styling to form fields
    # Implementation depends on UI design requirements
    pass

# One-liner utilities untuk common UI operations
get_widget_value = lambda ui_components, widget_name, default=None: getattr(ui_components.get(widget_name), 'value', default) if ui_components.get(widget_name) else default
set_widget_disabled = lambda ui_components, widget_name, disabled=True: setattr(ui_components[widget_name], 'disabled', disabled) if widget_name in ui_components else None
has_valid_api_key = lambda ui_components: bool(get_widget_value(ui_components, 'api_key_input', '').strip())
is_form_complete = lambda ui_components: all(get_widget_value(ui_components, field, '').strip() for field in ['workspace_input', 'project_input', 'version_input'])
get_download_config_from_ui = lambda ui_components: {
    'validate': get_widget_value(ui_components, 'validate_checkbox', True),
    'backup': get_widget_value(ui_components, 'backup_checkbox', False)
}
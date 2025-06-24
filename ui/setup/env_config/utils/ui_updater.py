"""
File: smartcash/ui/setup/env_config/utils/ui_updater.py
Deskripsi: Helper functions untuk update UI components
"""

from typing import Dict, Any, Optional
from smartcash.ui.setup.env_config.constants import STATUS_COLORS, UI_ELEMENTS

def update_status_panel(ui_components: Dict[str, Any], message: str, 
                       status_type: str = 'info') -> None:
    """🎨 Update status panel dengan color-coding"""
    if UI_ELEMENTS['status_panel'] not in ui_components:
        return
    
    color = STATUS_COLORS.get(status_type, STATUS_COLORS['neutral'])
    
    ui_components[UI_ELEMENTS['status_panel']].value = f"""
    <div style="padding: 10px; border-left: 4px solid {color}; background: rgba({_hex_to_rgb(color)}, 0.1);">
        <strong>{message}</strong>
    </div>
    """

def update_progress_bar(ui_components: Dict[str, Any], progress: int, 
                       message: str = "", is_error: bool = False) -> None:
    """📊 Update progress bar dan text"""
    progress = max(0, min(100, progress))
    color = STATUS_COLORS['error'] if is_error else STATUS_COLORS['success']
    
    # Update progress bar
    if UI_ELEMENTS['progress_bar'] in ui_components:
        ui_components[UI_ELEMENTS['progress_bar']].value = progress
    
    # Update progress text
    if UI_ELEMENTS['progress_text'] in ui_components:
        ui_components[UI_ELEMENTS['progress_text']].value = f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>{message}</span>
            <span style="color: {color}; font-weight: bold;">{progress}%</span>
        </div>
        """

def update_setup_button(ui_components: Dict[str, Any], enabled: bool = True, 
                       text: str = "Setup Environment") -> None:
    """🔘 Update setup button state"""
    if UI_ELEMENTS['setup_button'] not in ui_components:
        return
    
    button = ui_components[UI_ELEMENTS['setup_button']]
    button.disabled = not enabled
    button.description = text
    
    # Visual feedback
    if enabled:
        button.button_style = 'success'
        button.icon = 'play'
    else:
        button.button_style = 'warning' 
        button.icon = 'hourglass'

def update_summary_panel(ui_components: Dict[str, Any], summary_data: Dict[str, Any]) -> None:
    """📋 Update environment summary panel"""
    if UI_ELEMENTS['summary_panel'] not in ui_components:
        return
    
    summary_html = _generate_summary_html(summary_data)
    ui_components[UI_ELEMENTS['summary_panel']].value = summary_html

def append_to_log(ui_components: Dict[str, Any], message: str, 
                 level: str = 'info') -> None:
    """📝 Append message ke log accordion"""
    if UI_ELEMENTS['log_accordion'] not in ui_components:
        return
    
    log_widget = ui_components[UI_ELEMENTS['log_accordion']]
    timestamp = _get_timestamp()
    color = _get_log_color(level)
    
    new_entry = f'<div style="color: {color};">[{timestamp}] {message}</div>'
    
    # Append to existing log
    current_value = getattr(log_widget, 'value', '')
    log_widget.value = current_value + new_entry + '<br>'

def clear_all_ui(ui_components: Dict[str, Any]) -> None:
    """🧹 Clear semua UI components"""
    for element_id in UI_ELEMENTS.values():
        if element_id in ui_components:
            widget = ui_components[element_id]
            if hasattr(widget, 'value'):
                widget.value = '' if hasattr(widget.value, 'strip') else 0

def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string"""
    hex_color = hex_color.lstrip('#')
    return ', '.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))

def _get_timestamp() -> str:
    """Get formatted timestamp"""
    from datetime import datetime
    return datetime.now().strftime('%H:%M:%S')

def _get_log_color(level: str) -> str:
    """Get color untuk log level"""
    level_colors = {
        'info': STATUS_COLORS['info'],
        'success': STATUS_COLORS['success'],
        'warning': STATUS_COLORS['warning'],
        'error': STATUS_COLORS['error']
    }
    return level_colors.get(level, STATUS_COLORS['neutral'])

def _generate_summary_html(summary_data: Dict[str, Any]) -> str:
    """Generate HTML untuk summary panel"""
    items = []
    for key, value in summary_data.items():
        if isinstance(value, bool):
            icon = "✅" if value else "❌"
            items.append(f"<li>{icon} {key.replace('_', ' ').title()}</li>")
        else:
            items.append(f"<li>📋 {key.replace('_', ' ').title()}: {value}</li>")
    
    return f"<ul>{''.join(items)}</ul>"
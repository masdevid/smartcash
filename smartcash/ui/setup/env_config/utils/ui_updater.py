"""
File: smartcash/ui/setup/env_config/components/ui_updater.py
Deskripsi: Fixed UI updater dengan left/right positioning yang benar
"""

from typing import Dict, Any, Optional
from smartcash.ui.setup.env_config.constants import STATUS_COLORS, UI_ELEMENTS

def update_summary_panels(ui_components: Dict[str, Any], 
                         env_summary: str = None, 
                         system_info: str = None) -> None:
    """📋 Update dual summary panels dengan positioning yang benar"""
    
    # Update environment summary (kiri) - termasuk Google Drive status
    if env_summary and 'summary_panel_left' in ui_components:
        ui_components['summary_panel_left'].value = env_summary
    
    # Update system info (kanan) - specs system saja  
    if system_info and 'summary_panel_right' in ui_components:
        ui_components['summary_panel_right'].value = system_info

def update_status_panel(ui_components: Dict[str, Any], message: str, 
                       status_type: str = 'info') -> None:
    """🎨 Update status panel dengan color-coding"""
    if 'status_panel' not in ui_components:
        return
    
    color = STATUS_COLORS.get(status_type, STATUS_COLORS['neutral'])
    bg_color = _get_background_color(status_type)
    border_color = _get_border_color(status_type)
    
    ui_components['status_panel'].value = f"""
    <p style='color: {color}; background: {bg_color}; padding: 12px; margin: 8px 0; 
    border: 1px solid {border_color}; border-radius: 8px;'>{message}</p>
    """

def update_progress_bar(ui_components: Dict[str, Any], progress: int, 
                       message: str = "", is_error: bool = False) -> None:
    """📊 Update progress bar dan text"""
    progress = max(0, min(100, progress))
    
    # Update progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = progress
    
    # Update progress text dengan color coding
    if 'progress_text' in ui_components:
        color = STATUS_COLORS['error'] if is_error else STATUS_COLORS['success']
        weight = 'bold' if is_error or progress == 100 else 'normal'
        
        ui_components['progress_text'].value = f"""
        <span style='color: {color}; font-weight: {weight};'>{message}</span>
        """

def update_setup_button(ui_components: Dict[str, Any], enabled: bool = True, 
                       text: str = "Setup Environment") -> None:
    """🔘 Update setup button state"""
    if 'setup_button' not in ui_components:
        return
    
    button = ui_components['setup_button']
    button.disabled = not enabled
    button.description = text
    
    # Visual feedback
    if enabled:
        button.button_style = 'success'
        button.icon = 'play'
    else:
        button.button_style = 'warning' 
        button.icon = 'hourglass'

def _get_background_color(status_type: str) -> str:
    """Get background color untuk status type"""
    color_map = {
        'success': '#d4edda',
        'warning': '#fff3cd', 
        'error': '#f8d7da',
        'info': '#d1ecf1'
    }
    return color_map.get(status_type, '#f8f9fa')

def _get_border_color(status_type: str) -> str:
    """Get border color untuk status type"""
    color_map = {
        'success': '#c3e6cb',
        'warning': '#ffeaa7',
        'error': '#f5c6cb', 
        'info': '#bee5eb'
    }
    return color_map.get(status_type, '#dee2e6')

def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color ke RGB string"""
    hex_color = hex_color.lstrip('#')
    return ','.join(str(int(hex_color[i:i+2], 16)) for i in (0, 2, 4))
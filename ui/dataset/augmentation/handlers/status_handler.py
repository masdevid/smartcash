"""
File: smartcash/ui/dataset/augmentation/handlers/status_handler.py
Deskripsi: Handler status untuk augmentasi dataset
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import ALERT_STYLES, ICONS

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update status panel dengan pesan dan jenis status, menggunakan alert_utils standar.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan status
    """
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback jika alert_utils tidak tersedia
        if 'status_panel' in ui_components:
            style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
            bg_color = style.get('bg_color', '#d1ecf1')
            text_color = style.get('text_color', '#0c5460')
            border_color = style.get('border_color', '#0c5460') 
            icon = style.get('icon', ICONS.get(status_type, 'ℹ️'))
            
            ui_components['status_panel'].value = f"""
            <div style="padding: 10px; background-color: {bg_color}; 
                        color: {text_color}; margin: 10px 0; border-radius: 4px; 
                        border-left: 4px solid {border_color};">
                <p style="margin:5px 0">{icon} {message}</p>
            </div>
            """
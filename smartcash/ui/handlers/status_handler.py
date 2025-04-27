"""
File: smartcash/ui/handlers/status_handler.py
Deskripsi: Handler status yang konsisten untuk modul preprocessing dan augmentasi
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.constants import ALERT_STYLES

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update status panel UI dengan pesan dan tipe yang ditentukan.
    
    Args:
        ui_components: Dictionary komponen UI dengan kunci 'status_panel'
        status_type: Tipe pesan ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    status_panel = ui_components.get('status_panel')
    if not status_panel or not hasattr(status_panel, 'value'):
        return
    
    # Dapatkan style dari alert_styles
    style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    
    # Update HTML pada panel
    status_panel.value = f"""
    <div style="padding:10px; background-color:{style['bg_color']}; 
              color:{style['text_color']}; border-radius:4px; margin:5px 0;
              border-left:4px solid {style['text_color']};">
        <p style="margin:5px 0">{style['icon']} {message}</p>
    </div>"""

def create_status_panel(message: str = "", status_type: str = "info") -> widgets.HTML:
    """
    Buat komponen status panel untuk modul dengan styling yang konsisten.
    
    Args:
        message: Pesan awal untuk status
        status_type: Tipe status awal ('info', 'success', 'warning', 'error')
        
    Returns:
        Widget HTML berisi status panel
    """
    # Dapatkan style dari alert_styles
    style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    
    # Buat widget HTML
    return widgets.HTML(
        value=f"""
        <div style="padding:10px; background-color:{style['bg_color']}; 
                  color:{style['text_color']}; border-radius:4px; margin:5px 0;
                  border-left:4px solid {style['text_color']};">
            <p style="margin:5px 0">{style['icon']} {message}</p>
        </div>"""
    )
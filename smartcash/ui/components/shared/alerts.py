"""
File: smartcash/ui/components/shared/alerts.py
Deskripsi: Komponen UI untuk alerts, info boxes, dan status indicators
"""

import ipywidgets as widgets
from IPython.display import HTML
from typing import Optional, Union

from smartcash.ui.utils.constants import ALERT_STYLES, COLORS

def create_status_indicator(status: str, message: str) -> HTML:
    """
    Buat indikator status dengan style yang sesuai.
    
    Args:
        status: Jenis status ('info', 'success', 'warning', 'error')
        message: Pesan status
        
    Returns:
        Widget HTML berisi indikator status
    """
    style_config = ALERT_STYLES.get(status, ALERT_STYLES['info'])
    
    return HTML(f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: {COLORS['light']};">
        <span style="color: {style_config['text_color']}; font-weight: bold;"> 
            {style_config['icon']} {message}
        </span>
    </div>
    """)

def create_info_alert(message: str, alert_type: str = 'info', icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat alert box dengan style yang sesuai.
    
    Args:
        message: Pesan alert
        alert_type: Jenis alert ('info', 'success', 'warning', 'error')
        icon: Emoji icon opsional, jika tidak diisi akan menggunakan icon default
        
    Returns:
        Widget HTML berisi alert
    """
    style_config = ALERT_STYLES.get(alert_type, ALERT_STYLES['info'])
    icon_str = icon if icon else style_config['icon']
    
    alert_html = f"""
    <div style="padding: 10px; 
                background-color: {style_config['bg_color']}; 
                color: {style_config['text_color']}; 
                border-left: 4px solid {style_config['border_color']}; 
                border-radius: 5px; 
                margin: 10px 0;">
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 10px; font-size: 1.2em;">{icon_str}</div>
            <div>{message}</div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=alert_html)

def create_info_box(title: str, content: str, style: str = 'info', 
                  icon: Optional[str] = None, collapsed: bool = False) -> Union[widgets.HTML, widgets.Accordion]:
    """
    Buat info box yang dapat di-collapse.
    
    Args:
        title: Judul info box
        content: Konten HTML info box
        style: Jenis style ('info', 'success', 'warning', 'error')
        icon: Emoji icon opsional
        collapsed: Apakah info box collapsed secara default
        
    Returns:
        Widget HTML atau Accordion berisi info box
    """
    style_config = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    icon_to_use = icon if icon else style_config['icon']
    title_with_icon = f"{icon_to_use} {title}"
    
    if collapsed:
        # Gunakan Accordion jika perlu collapsible
        content_widget = widgets.HTML(value=content)
        accordion = widgets.Accordion([content_widget])
        accordion.set_title(0, title_with_icon)
        accordion.selected_index = None
        return accordion
    else:
        # Gunakan HTML biasa jika tidak perlu collapsible
        box_html = f"""
        <div style="padding: 10px; background-color: {style_config['bg_color']}; 
                 border-left: 4px solid {style_config['border_color']}; 
                 color: {style_config['text_color']}; margin: 10px 0; border-radius: 4px;">
            <h4 style="margin-top: 0; color: inherit;">{title_with_icon}</h4>
            {content}
        </div>
        """
        return widgets.HTML(value=box_html)
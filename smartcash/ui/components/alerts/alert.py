"""
File: smartcash/ui/components/alerts/alert.py
Deskripsi: Komponen dasar untuk alert
"""

from typing import Optional
import ipywidgets as widgets
from IPython.display import HTML

from .constants import ALERT_STYLES, COLORS

def create_alert(
    message: str, 
    alert_type: str = 'info', 
    title: Optional[str] = None,
    icon: Optional[str] = None,
    closable: bool = False
) -> widgets.HTML:
    """Buat komponen alert yang dapat digunakan kembali.
    
    Args:
        message: Pesan yang akan ditampilkan
        alert_type: Tipe alert ('info', 'success', 'warning', 'error')
        title: Judul opsional untuk alert
        icon: Ikon opsional (default: ikon bawaan berdasarkan tipe)
        closable: Apakah alert bisa ditutup
        
    Returns:
        Widget HTML yang berisi alert
    """
    style = ALERT_STYLES.get(alert_type.lower(), ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    title_text = title or style['title']
    
    close_button = ""
    if closable:
        close_button = """
        <button type="button" class="close" data-dismiss="alert" aria-label="Close" 
                style="float: right; background: none; border: none; cursor: pointer;
                       font-size: 1.5em; line-height: 1; padding: 0 0.5em;"
                onclick="this.parentElement.style.display='none';">
            &times;
        </button>
        """
    
    html_content = f"""
    <div class="alert alert-{alert_type}" 
         style="padding: 1em; margin: 0.5em 0; 
                border-left: 4px solid {style['border_color']};
                background-color: {style['bg_color']};
                color: {style['text_color']};
                border-radius: 4px;
                position: relative;">
        {close_button}
        <div style="display: flex; align-items: flex-start;">
            <div style="margin-right: 0.75em; font-size: 1.2em;">{icon_str}</div>
            <div>
                {f'<div style="font-weight: bold; margin-bottom: 0.5em;">{title_text}</div>' if title_text else ''}
                <div style="margin: 0.25em 0;">{message}</div>
            </div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=html_content)

def create_alert_html(
    message: str, 
    alert_type: str = 'info', 
    title: Optional[str] = None,
    icon: Optional[str] = None
) -> HTML:
    """Buat HTML untuk alert (tanpa widget).
    
    Args:
        message: Pesan yang akan ditampilkan
        alert_type: Tipe alert ('info', 'success', 'warning', 'error')
        title: Judul opsional untuk alert
        icon: Ikon opsional (default: ikon bawaan berdasarkan tipe)
        
    Returns:
        Objek HTML yang berisi alert
    """
    alert_widget = create_alert(message, alert_type, title, icon)
    return HTML(alert_widget.value)

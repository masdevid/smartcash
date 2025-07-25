"""
File: smartcash/ui/components/alerts/status_indicator.py
Deskripsi: Komponen untuk menampilkan indikator status
"""

from typing import Optional
from IPython.display import HTML
import ipywidgets as widgets

from .constants import ALERT_STYLES, COLORS

def create_status_indicator(status: str, message: str, icon: Optional[str] = None) -> HTML:
    """Buat indikator status yang distilisasi.
    
    Args:
        status: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan status
        icon: Ikon opsional (default: ikon bawaan berdasarkan tipe)
        
    Returns:
        Objek HTML dengan indikator status
    """
    style = ALERT_STYLES.get(status.lower(), ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    
    html_content = (
        f'<div style="margin: 5px 0; padding: 8px 12px; '
        f'border-radius: 4px; background-color: {COLORS["light"]};">'
        f'<span style="color: {style["text_color"]}; font-weight: bold;">'
        f'{icon_str} {message}'
        f'</span></div>'
    )
    return HTML(html_content)

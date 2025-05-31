"""
File: smartcash/ui/components/status_panel.py
Deskripsi: Komponen status panel yang reusable dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_status_panel(message: str = "", status_type: str = "info", layout: Optional[Dict[str, Any]] = None) -> widgets.HTML:
    """Buat status panel yang bisa digunakan kembali dengan one-liner style."""
    from smartcash.ui.utils.constants import ALERT_STYLES
    style_info = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    bg_color, text_color, icon = style_info['bg_color'], style_info['text_color'], style_info['icon']
    html_content = f"""<div style="padding:10px; background-color:{bg_color}; color:{text_color}; border-radius:4px; margin:5px 0; border-left:4px solid {text_color};"><p style="margin:5px 0">{icon} {message}</p></div>"""
    default_layout = {'width': '100%', 'margin': '10px 0'}
    layout and default_layout.update(layout)
    return widgets.HTML(value=html_content, layout=widgets.Layout(**default_layout))

def update_status_panel(panel: widgets.HTML, message: str, status_type: str = "info") -> None:
    """Update status panel yang sudah ada dengan one-liner style."""
    from smartcash.ui.utils.constants import ALERT_STYLES
    style_info = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    bg_color, text_color, icon = style_info['bg_color'], style_info['text_color'], style_info['icon']
    setattr(panel, 'value', f"""<div style="padding:10px; background-color:{bg_color}; color:{text_color}; border-radius:4px; margin:5px 0; border-left:4px solid {text_color};"><p style="margin:5px 0">{icon} {message}</p></div>""")

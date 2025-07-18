"""
File: smartcash/ui/components/alerts/info_box.py
Deskripsi: Komponen untuk membuat info box yang bisa dilipat
"""

from typing import Optional, Union
import ipywidgets as widgets
from IPython.display import HTML

from .constants import ALERT_STYLES

def create_info_box(
    title: str, 
    content: str, 
    style: str = 'info', 
    icon: Optional[str] = None,
    collapsed: bool = False
) -> Union[widgets.HTML, widgets.Accordion]:
    """Buat info box yang bisa dilipat.
    
    Args:
        title: Judul info box
        content: Konten HTML untuk info box
        style: Tipe gaya ('info', 'success', 'warning', 'error')
        icon: Ikon opsional (default: ikon bawaan berdasarkan tipe)
        collapsed: Apakah box dalam keadaan terlipat secara default
        
    Returns:
        Widget HTML atau Accordion dengan info box
    """
    style = ALERT_STYLES.get(style.lower(), ALERT_STYLES['info'])
    icon_str = icon or style['icon']
    
    # Buat header dengan ikon dan judul
    header = f"""
    <div style="display: flex; align-items: center;">
        <span style="margin-right: 8px; font-size: 1.1em;">{icon_str}</span>
        <span>{title}</span>
    </div>
    """
    
    # Jika tidak perlu accordion (sederhana)
    if not collapsed:
        html_content = f"""
        <div style="margin: 10px 0; border: 1px solid {border_color};
                   border-radius: 4px; overflow: hidden;">
            <div style="padding: 8px 12px; background-color: {bg_color};
                      color: {text_color}; font-weight: bold;">
                {header}
            </div>
            <div style="padding: 12px; background-color: white;
                      border-top: 1px solid {border_color};">
                {content}
            </div>
        </div>
        """.format(
            bg_color=style['bg_color'],
            border_color=style['border_color'],
            text_color=style['text_color'],
            header=header.strip(),
            content=content
        )
        return widgets.HTML(value=html_content)
    
    # Buat accordion untuk versi yang bisa dilipat
    content_widget = widgets.HTML(value=content)
    accordion = widgets.Accordion(children=[content_widget])
    accordion.set_title(0, f"{icon_str} {title}")
    
    # Terapkan gaya ke accordion
    accordion.add_class('alert-info-box')
    accordion.layout.border = f"1px solid {style['border_color']}"
    accordion.layout.border_radius = '4px'
    accordion.layout.overflow = 'hidden'
    accordion.layout.margin = '10px 0'
    
    # Gaya untuk header accordion
    accordion.style.header_background_color = style['bg_color']
    accordion.style.header_text_color = style['text_color']
    
    return accordion

"""
File: smartcash/ui/utils/headers.py
Deskripsi: Komponen header dan section title yang menggunakan ui_helpers untuk konsistensi
"""
from typing import Optional
import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS, ICONS


def create_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> widgets.HTML:
    """
    Buat komponen header dengan style konsisten.
    
    Args:
        title: Judul header
        description: Deskripsi opsional
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi header
    """
    # Tambahkan ikon jika disediakan
    title_with_icon = f"{icon} {title}" if icon else title
    
    header_html = f"""
    <div style="background-color: {COLORS['header_bg']}; padding: 15px; color: black; 
            border-radius: 5px; margin-bottom: 15px; border-left: 5px solid {COLORS['primary']};">
        <h2 style="color: {COLORS['dark']}; margin-top: 0;">{title_with_icon}</h2>
    """
    
    if description:
        header_html += f'<p style="color: black; margin-bottom: 0;">{description}</p>'
    
    header_html += "</div>"
    
    return widgets.HTML(value=header_html)


def create_section_title(title: str, icon: Optional[str] = "") -> widgets.HTML:
    """
    Buat judul section dengan style konsisten.
    
    Args:
        title: Judul section
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi judul section
    """
    return widgets.HTML(f"""
    <h3 style="color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;">
        {icon} {title}
    </h3>
    """)

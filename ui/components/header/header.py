"""
File: smartcash/ui/components/header/header.py
Deskripsi: Komponen header untuk UI yang konsisten
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
    try:
        # Default colors
        colors = {
            'header_bg': '#f0f8ff',
            'primary': '#3498db',
            'dark': '#212529'
        }
        
        # Update with COLORS if available
        if COLORS:
            colors.update({
                'header_bg': COLORS.get('header_bg', colors['header_bg']),
                'primary': COLORS.get('primary', colors['primary']),
                'dark': COLORS.get('dark', colors['dark'])
            })
        
        # Add icon if provided
        title_with_icon = f"{icon} {title}" if icon else title
        
        header_html = f"""
        <div style="background-color: {bg_color}; padding: 15px; color: black; 
                border-radius: 5px; margin-bottom: 15px; border-left: 5px solid {border_color};">
            <h2 style="color: {text_color}; margin-top: 0;">{title_with_icon}</h2>
        """.format(
            bg_color=colors['header_bg'],
            border_color=colors['primary'],
            text_color=colors['dark'],
            title_with_icon=title_with_icon
        )
        
        if description:
            header_html += f'<p style="color: black; margin-bottom: 0;">{description}</p>'
        
        header_html += "</div>"
        
        return widgets.HTML(value=header_html)
        
    except Exception as e:
        # Fallback minimal header
        title_with_icon = f"{icon} {title}" if icon else title
        return widgets.HTML(
            f'<div style="padding:15px;margin-bottom:15px;border-left:5px solid #3498db">'
            f'<h2 style="margin-top:0">{title_with_icon}</h2>'
            f'<p style="margin-bottom:0">{description or ""}</p>'
            '</div>'
        )


def create_section_title(title: str, icon: Optional[str] = "") -> widgets.HTML:
    """
    Buat judul section dengan style konsisten.
    
    Args:
        title: Judul section
        icon: Emoji icon opsional
        
    Returns:
        Widget HTML berisi judul section
    """
    try:
        # Default color
        primary_color = '#007bff'
        
        # Use COLORS if available
        if COLORS and 'primary' in COLORS:
            primary_color = COLORS['primary']
            
        title_with_icon = f"{icon} {title}" if icon else title
        
        return widgets.HTML(
            f'<h3 style="color:{primary_color};'
            'margin-top:20px;margin-bottom:10px;border-bottom:1px solid #eee;'
            'padding-bottom:5px;font-weight:500">'
            f'{title_with_icon}</h3>'
        )
    except Exception:
        # Fallback minimal section title
        title_with_icon = f"{icon} {title}" if icon else title
        return widgets.HTML(f'<h3 style="color:#007bff;margin:20px 0 10px">{title_with_icon}</h3>')

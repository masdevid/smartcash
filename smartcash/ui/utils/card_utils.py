"""
File: smartcash/ui/utils/card_utils.py
Deskripsi: Utilitas untuk membuat komponen kartu UI
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets
from IPython.display import HTML

from smartcash.ui.utils.constants import COLORS

def create_card_html(
    title: str, 
    value: str, 
    icon: str, 
    color: str, 
    description: Optional[str] = None,
    footer: Optional[str] = None,
    width: str = "250px",
    height: str = "auto"
) -> str:
    """
    Membuat HTML untuk kartu statistik.
    
    Args:
        title: Judul kartu
        value: Nilai utama yang ditampilkan
        icon: Ikon emoji
        color: Warna aksen kartu
        description: Deskripsi opsional
        footer: Footer opsional
        width: Lebar kartu
        height: Tinggi kartu
        
    Returns:
        String HTML kartu
    """
    card_html = f'''
    <div style="
        width: {width}; 
        height: {height}; 
        background-color: {COLORS['card']}; 
        border-radius: 8px; 
        padding: 15px; 
        margin: 10px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-top: 3px solid {color};
    ">
        <div style="font-size: 1.5em; margin-bottom: 5px;">{icon}</div>
        <div style="font-size: 0.9em; color: {COLORS['dark']}; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 1.4em; font-weight: bold; color: {color}; margin-bottom: 5px;">{value}</div>
    '''
    
    if description:
        card_html += f'<div style="font-size: 0.8em; color: {COLORS["muted"]}; margin-bottom: 5px;">{description}</div>'
    
    if footer:
        card_html += f'<div style="font-size: 0.8em; color: {COLORS["muted"]}; margin-top: 10px; padding-top: 5px; border-top: 1px solid {COLORS["border"]};">{footer}</div>'
    
    card_html += '</div>'
    
    return card_html

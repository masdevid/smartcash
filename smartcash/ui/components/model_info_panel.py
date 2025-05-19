"""
File: smartcash/ui/components/model_info_panel.py
Deskripsi: Komponen shared untuk panel informasi model
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List
from smartcash.ui.utils.constants import ICONS, COLORS

def create_model_info_panel(
    title: str = "Informasi Model",
    min_height: str = "100px",
    width: str = "100%",
    icon: str = "info"
) -> Dict[str, Any]:
    """
    Buat panel informasi model yang dapat digunakan di berbagai modul.
    
    Args:
        title: Judul panel informasi
        min_height: Tinggi minimal panel
        width: Lebar panel
        icon: Ikon untuk judul panel
        
    Returns:
        Dictionary berisi komponen panel informasi
    """
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat panel untuk informasi model
    info_panel = widgets.Output(
        layout=widgets.Layout(
            width=width, 
            min_height=min_height,
            border='1px solid #ddd',
            padding='10px',
            margin='5px 0px'
        )
    )
    
    # Buat header untuk panel
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 0; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat container untuk panel
    container = widgets.VBox([
        header,
        info_panel
    ], layout=widgets.Layout(margin='10px 0px'))
    
    return {
        'container': container,
        'info_panel': info_panel,
        'header': header
    }

"""
File: smartcash/ui/components/feature_checkbox_group.py
Deskripsi: Komponen shared untuk grup checkbox fitur
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_feature_checkbox_group(
    features: List[Tuple[str, bool]] = None,
    title: str = "Fitur Optimasi",
    description: str = None,
    width: str = "100%",
    icon: str = "settings"
) -> Dict[str, Any]:
    """
    Buat grup checkbox untuk fitur yang dapat digunakan di berbagai modul.
    
    Args:
        features: List tuple (deskripsi, nilai default) untuk checkbox
        title: Judul grup checkbox
        description: Deskripsi tambahan (opsional)
        width: Lebar komponen
        icon: Ikon untuk judul
        
    Returns:
        Dictionary berisi komponen grup checkbox
    """
    # Default features jika tidak ada yang diberikan
    if features is None:
        features = [
            ("Fitur 1", False),
            ("Fitur 2", False),
            ("Fitur 3", False)
        ]
    
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat header untuk grup
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat deskripsi jika ada
    description_widget = None
    if description:
        description_widget = widgets.HTML(
            value=f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>"
        )
    
    # Buat checkbox untuk setiap fitur
    checkboxes = {}
    checkbox_widgets = []
    
    for desc, default_value in features:
        checkbox = widgets.Checkbox(
            value=default_value,
            description=desc,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width=width)
        )
        checkbox_widgets.append(checkbox)
        # Gunakan deskripsi sebagai key dengan format snake_case
        key = desc.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        checkboxes[key] = checkbox
    
    # Buat container untuk grup checkbox
    widgets_list = [header]
    if description_widget:
        widgets_list.append(description_widget)
    widgets_list.extend(checkbox_widgets)
    
    container = widgets.VBox(
        widgets_list,
        layout=widgets.Layout(
            margin='10px 0px',
            padding='10px',
            border='1px solid #eee',
            border_radius='4px'
        )
    )
    
    return {
        'container': container,
        'checkboxes': checkboxes,
        'header': header
    }

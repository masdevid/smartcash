"""
File: smartcash/ui/components/validation_options.py
Deskripsi: Komponen shared untuk opsi validasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_validation_options(
    title: str = "Opsi Validasi",
    description: str = "Pilih opsi validasi untuk dataset",
    options: List[Tuple[str, str, bool]] = None,
    width: str = "100%",
    icon: str = "validation"
) -> Dict[str, Any]:
    """
    Buat komponen opsi validasi yang dapat digunakan di berbagai modul.
    
    Args:
        title: Judul komponen
        description: Deskripsi komponen
        options: List tuple (label, value, default) untuk opsi validasi
        width: Lebar komponen
        icon: Ikon untuk judul
        
    Returns:
        Dictionary berisi komponen opsi validasi
    """
    # Default options jika tidak disediakan
    if options is None:
        options = [
            ("Validasi ukuran gambar", "validate_image_size", True),
            ("Validasi format gambar", "validate_image_format", True),
            ("Validasi anotasi", "validate_annotations", True),
            ("Validasi duplikasi", "validate_duplicates", False)
        ]
    
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat header untuk komponen
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat deskripsi jika ada
    description_widget = None
    if description:
        description_widget = widgets.HTML(
            value=f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>"
        )
    
    # Buat checkbox untuk setiap opsi
    checkboxes = {}
    checkbox_widgets = []
    
    for label, value, default in options:
        # Buat nama key dengan format snake_case
        key = value.lower().replace(" ", "_")
        
        # Buat checkbox
        checkbox = widgets.Checkbox(
            value=default,
            description=label,
            layout=widgets.Layout(width=width),
            style={'description_width': 'initial'}
        )
        
        # Tambahkan ke dictionary dan list
        checkboxes[key] = checkbox
        checkbox_widgets.append(checkbox)
    
    # Buat container untuk komponen
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

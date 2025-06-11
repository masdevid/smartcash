"""
File: smartcash/ui/components/validation_options.py
Deskripsi: Komponen shared untuk opsi validasi dataset dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_validation_options(title: str = "Opsi Validasi", description: str = "Pilih opsi validasi untuk dataset",
                             options: List[Tuple[str, str, bool]] = None, width: str = "100%", icon: str = "validation") -> Dict[str, Any]:
    """Buat komponen opsi validasi dengan one-liner style."""
    options = options or [("Validasi ukuran gambar", "validate_image_size", True), ("Validasi format gambar", "validate_image_format", True),
                         ("Validasi anotasi", "validate_annotations", True), ("Validasi duplikasi", "validate_duplicates", False)]
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    
    header = widgets.HTML(f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    description_widget = widgets.HTML(f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>") if description else None
    
    checkboxes = {value.lower().replace(" ", "_"): widgets.Checkbox(value=default, description=label, layout=widgets.Layout(width=width), 
                                                                   style={'description_width': 'initial'}) for label, value, default in options}
    checkbox_widgets = list(checkboxes.values())
    
    widgets_list = [header] + ([description_widget] if description_widget else []) + checkbox_widgets
    container = widgets.VBox(widgets_list, layout=widgets.Layout(margin='10px 0px', padding='10px', border='1px solid #eee', border_radius='4px'))
    
    return {'container': container, 'checkboxes': checkboxes, 'header': header}
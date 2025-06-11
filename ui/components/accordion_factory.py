"""
File: smartcash/ui/components/accordion_factory.py
Deskripsi: Utilitas untuk membuat komponen accordion dengan one-liner style yang lebih ringkas
"""

import ipywidgets as widgets
from typing import List, Tuple, Any

def create_accordion(accordion_items: List[Tuple[str, Any]], selected_index: int = None) -> widgets.Accordion:
    """Membuat widget accordion dengan pendekatan DRY one-liner."""
    # Pastikan semua item adalah widget, bukan string
    section_widgets, section_titles = [], []
    
    if accordion_items:
        for title, widget in accordion_items:
            if not isinstance(widget, widgets.Widget):
                # Konversi ke widget jika bukan widget
                widget = widgets.HTML(str(widget))
            section_widgets.append(widget)
            section_titles.append(title)
    
    # Buat accordion dengan children yang sudah divalidasi
    accordion = widgets.Accordion(children=section_widgets)
    
    # Set titles
    for i, title in enumerate(section_titles):
        accordion.set_title(i, title)
    
    # Set selected index
    if selected_index is not None and 0 <= selected_index < len(section_titles):
        accordion.selected_index = selected_index
    
    return accordion
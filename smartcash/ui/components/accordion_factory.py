"""
File: smartcash/ui/components/accordion_factory.py
Deskripsi: Utilitas untuk membuat komponen accordion dengan pendekatan DRY
"""

import ipywidgets as widgets
from typing import List, Tuple, Any, Union, Dict

def create_accordion(accordion_items: List[Tuple[str, Any]], selected_index: int = None) -> widgets.Accordion:
    """
    Membuat widget accordion dengan pendekatan DRY.
    
    Args:
        accordion_items: List berisi tuple (section_title, section_content)
        selected_index: Indeks section yang terbuka (opsional)
        
    Returns:
        Widget Accordion yang telah dikonfigurasi
    """
    # Buat widget untuk setiap section
    section_widgets = []
    section_titles = []
    
    # Tambahkan setiap item ke accordion
    for title, content in accordion_items:
        section_widgets.append(content)
        section_titles.append(title)
    
    # Buat accordion
    accordion = widgets.Accordion(children=section_widgets)
    
    # Set judul section
    for i, title in enumerate(section_titles):
        accordion.set_title(i, title)
    
    # Set section yang terbuka jika ditentukan
    if selected_index is not None and 0 <= selected_index < len(section_titles):
        accordion.selected_index = selected_index
    
    return accordion

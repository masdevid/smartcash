"""
File: smartcash/ui/components/accordion_factory.py
Deskripsi: Utilitas untuk membuat komponen accordion dengan one-liner style yang lebih ringkas
"""

import ipywidgets as widgets
from typing import List, Tuple, Any

def create_accordion(accordion_items: List[Tuple[str, Any]], selected_index: int = None) -> widgets.Accordion:
    """Membuat widget accordion dengan pendekatan DRY one-liner."""
    section_widgets, section_titles = zip(*accordion_items) if accordion_items else ([], [])
    accordion = widgets.Accordion(children=list(section_widgets))
    [accordion.set_title(i, title) for i, title in enumerate(section_titles)]
    accordion.selected_index = selected_index if selected_index is not None and 0 <= selected_index < len(section_titles) else None
    return accordion
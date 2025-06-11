"""
File: smartcash/ui/components/tab_factory.py
Deskripsi: Utilitas untuk membuat komponen tab dengan one-liner style
"""

import ipywidgets as widgets
from typing import List, Tuple, Any

def create_tab_widget(tab_items: List[Tuple[str, Any]]) -> widgets.Tab:
    """Membuat widget tab dengan one-liner style."""
    # Pastikan semua item adalah widget, bukan string
    tab_widgets, tab_titles = [], []
    
    if tab_items:
        for title, widget in tab_items:
            if not isinstance(widget, widgets.Widget):
                # Konversi ke widget jika bukan widget
                widget = widgets.HTML(str(widget))
            tab_widgets.append(widget)
            tab_titles.append(title)
    
    # Buat tab dengan children yang sudah divalidasi
    tabs = widgets.Tab(children=tab_widgets)
    
    # Set titles
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
        
    return tabs
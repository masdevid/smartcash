"""
File: smartcash/ui/components/tab_factory.py
Deskripsi: Utilitas untuk membuat komponen tab dengan one-liner style
"""

import ipywidgets as widgets
from typing import List, Tuple, Any

def create_tab_widget(tab_items: List[Tuple[str, Any]]) -> widgets.Tab:
    """Membuat widget tab dengan one-liner style."""
    tab_widgets, tab_titles = zip(*tab_items) if tab_items else ([], [])
    tabs = widgets.Tab(children=list(tab_widgets))
    [tabs.set_title(i, title) for i, title in enumerate(tab_titles)]
    return tabs
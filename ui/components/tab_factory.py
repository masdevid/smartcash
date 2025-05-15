"""
File: smartcash/ui/components/tab_factory.py
Deskripsi: Utilitas untuk membuat komponen tab dengan pendekatan DRY
"""

import ipywidgets as widgets
from typing import List, Tuple, Any, Union, Dict

def create_tab_widget(tab_items: List[Tuple[str, Any]]) -> widgets.Tab:
    """
    Membuat widget tab dengan pendekatan DRY.
    
    Args:
        tab_items: List berisi tuple (tab_title, tab_content)
        
    Returns:
        Widget Tab yang telah dikonfigurasi
    """
    # Buat widget untuk setiap tab
    tab_widgets = []
    tab_titles = []
    
    # Tambahkan setiap item ke tab
    for title, content in tab_items:
        tab_widgets.append(content)
        tab_titles.append(title)
    
    # Buat tabs
    tabs = widgets.Tab(children=tab_widgets)
    
    # Set judul tab
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
    
    return tabs

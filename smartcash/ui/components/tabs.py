"""
File: smartcash/ui/components/tabs.py
Deskripsi: Modul untuk membuat dan mengelola komponen tab dengan one-liner style
"""

from typing import List, Tuple, Any
import ipywidgets as widgets

def create_tab_widget(tab_items: List[Tuple[str, Any]]) -> widgets.Tab:
    """
    Membuat widget tab dengan one-liner style.
    
    Args:
        tab_items: Daftar tuple berisi (judul_tab, widget_konten)
        
    Returns:
        widgets.Tab: Widget tab yang sudah dikonfigurasi
    """
    # Validasi dan konversi input ke widget jika diperlukan
    tab_widgets, tab_titles = [], []
    
    if tab_items:
        for title, widget in tab_items:
            if not isinstance(widget, widgets.Widget):
                widget = widgets.HTML(str(widget))
            tab_widgets.append(widget)
            tab_titles.append(title)
    
    # Buat tab dengan children yang sudah divalidasi
    tabs = widgets.Tab(children=tab_widgets)
    
    # Set judul untuk setiap tab
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
        
    return tabs

def create_tabs(tabs_list: List[Tuple[str, Any]]) -> widgets.Tab:
    """
    Alias untuk create_tab_widget dengan nama yang lebih deskriptif.
    
    Args:
        tabs_list: Daftar tuple berisi (judul_tab, widget_konten)
        
    Returns:
        widgets.Tab: Widget tab yang sudah dikonfigurasi
    """
    return create_tab_widget(tabs_list)

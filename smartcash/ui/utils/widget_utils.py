"""
File: smartcash/ui/utils/widget_utils.py
Deskripsi: Utilitas untuk manipulasi widget UI dengan penanganan error
"""

from typing import Any, Optional
import ipywidgets as widgets

def safe_update_widget(widget: Any, value: Any) -> bool:
    """
    Update nilai widget dengan penanganan error.
    
    Args:
        widget: Widget yang akan diupdate
        value: Nilai baru untuk widget
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        widget.value = value
        return True
    except Exception as e:
        print(f"⚠️ Error update widget: {e}")
        return False

def create_widget_group(title: str, widgets_list: list, collapsed: bool = False) -> widgets.VBox:
    """
    Buat grup widget dengan judul.
    
    Args:
        title: Judul grup
        widgets_list: List widget yang akan dikelompokkan
        collapsed: Apakah grup awalnya collapsed
        
    Returns:
        VBox berisi grup widget
    """
    from smartcash.ui.utils.constants import COLORS
    
    header = widgets.HTML(
        f"<h4 style='color:{COLORS['dark']}; margin-top:10px; margin-bottom:5px;'>{title}</h4>"
    )
    
    if collapsed:
        accordion = widgets.Accordion(children=[widgets.VBox(widgets_list)])
        accordion.set_title(0, title)
        return widgets.VBox([accordion])
    else:
        return widgets.VBox([header, *widgets_list])

def register_observers(component: Any, callback: callable, names: str = 'value') -> None:
    """
    Daftarkan observer untuk komponen dan semua anaknya.
    
    Args:
        component: Komponen UI
        callback: Fungsi callback
        names: Nama event yang akan diobservasi
    """
    if hasattr(component, 'children'):
        for child in component.children:
            register_observers(child, callback, names)
    else:
        component.observe(callback, names=names)

def unregister_observers(component: Any, callback: callable, names: str = 'value') -> None:
    """
    Hapus observer dari komponen dan semua anaknya.
    
    Args:
        component: Komponen UI
        callback: Fungsi callback
        names: Nama event yang diobservasi
    """
    if hasattr(component, 'children'):
        for child in component.children:
            unregister_observers(child, callback, names)
    else:
        component.unobserve(callback, names=names)

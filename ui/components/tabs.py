"""
File: smartcash/ui/components/tabs.py
Deskripsi: Komponen tabs untuk UI dengan one-liner style
"""

from typing import List, Tuple, Any
import ipywidgets as widgets
from smartcash.ui.components.tab_factory import create_tab_widget

def create_tabs(tabs_list: List[Tuple[str, Any]]) -> widgets.Tab:
    """Buat komponen tabs dengan one-liner style konsisten."""
    return create_tab_widget(tabs_list)

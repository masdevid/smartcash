"""
File: smartcash/ui/components/info_accordion.py
Deskripsi: Komponen accordion untuk menampilkan informasi dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS

def create_info_accordion(title: str = "Informasi", content: Any = None, icon: str = "info", open_by_default: bool = False) -> Dict[str, Any]:
    """Membuat accordion untuk menampilkan informasi dengan one-liner style."""
    icon_display = ICONS.get(icon, "ℹ️")
    content = content or widgets.HTML("<div style='padding: 10px; background-color: #f8f9fa;'><p>Tidak ada informasi yang tersedia.</p></div>")
    accordion = widgets.Accordion(children=[content])
    accordion.set_title(0, f"{icon_display} {title}"), setattr(accordion, 'selected_index', 0 if open_by_default else None)
    container = widgets.VBox([accordion], layout=widgets.Layout(width='100%', margin='10px 0', padding='0'))
    return {'accordion': accordion, 'container': container, 'content': content, 'title': title}

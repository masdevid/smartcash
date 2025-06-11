
"""
File: smartcash/ui/components/model_info_panel.py
Deskripsi: Komponen shared untuk panel informasi model dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS

def create_model_info_panel(title: str = "Informasi Model", min_height: str = "100px", width: str = "100%", icon: str = "info") -> Dict[str, Any]:
    """Buat panel informasi model dengan one-liner style."""
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    info_panel = widgets.Output(layout=widgets.Layout(width=width, min_height=min_height, border='1px solid #ddd', padding='10px', margin='5px 0px'))
    header = widgets.HTML(f"<h4 style='margin-top: 0; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    container = widgets.VBox([header, info_panel], layout=widgets.Layout(margin='10px 0px'))
    return {'container': container, 'info_panel': info_panel, 'header': header}


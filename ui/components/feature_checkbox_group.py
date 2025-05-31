
"""
File: smartcash/ui/components/feature_checkbox_group.py
Deskripsi: Komponen shared untuk grup checkbox fitur dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_feature_checkbox_group(features: List[Tuple[str, bool]] = None, title: str = "Fitur Optimasi",
                                 description: str = None, width: str = "100%", icon: str = "settings") -> Dict[str, Any]:
    """Buat grup checkbox untuk fitur dengan one-liner style."""
    features = features or [("Fitur 1", False), ("Fitur 2", False), ("Fitur 3", False)]
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    
    header = widgets.HTML(f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    description_widget = widgets.HTML(f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>") if description else None
    
    checkboxes = {desc.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_'): 
                  widgets.Checkbox(value=default_value, description=desc, style={'description_width': 'initial'}, layout=widgets.Layout(width=width))
                  for desc, default_value in features}
    checkbox_widgets = list(checkboxes.values())
    
    widgets_list = [header] + ([description_widget] if description_widget else []) + checkbox_widgets
    container = widgets.VBox(widgets_list, layout=widgets.Layout(margin='10px 0px', padding='10px', border='1px solid #eee', border_radius='4px'))
    
    return {'container': container, 'checkboxes': checkboxes, 'header': header}

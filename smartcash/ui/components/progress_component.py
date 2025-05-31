"""
File: smartcash/ui/components/progress_component.py
Deskripsi: Komponen progress tracking dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS

def create_progress_component(module_name: str = "processing") -> Dict[str, Any]:
    """Membuat komponen progress tracking dengan one-liner style."""
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, description='Total:', bar_style='info', orientation='horizontal', 
                                      layout=widgets.Layout(visibility='hidden', width='100%'))
    current_progress = widgets.IntProgress(value=0, min=0, max=100, description='Step:', bar_style='info', orientation='horizontal',
                                          layout=widgets.Layout(visibility='hidden', width='100%'))
    overall_label, step_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0')), widgets.HTML("", layout=widgets.Layout(margin='2px 0'))
    progress_container = widgets.VBox([widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
                                      progress_bar, overall_label, current_progress, step_label])
    return {'container': progress_container, 'progress_bar': progress_bar, 'current_progress': current_progress, 'overall_label': overall_label, 'step_label': step_label}

def show_progress_component(ui_components: Dict[str, Any], show: bool = True) -> None:
    """Menampilkan atau menyembunyikan komponen progress dengan one-liner."""
    visibility = 'visible' if show else 'hidden'
    [setattr(ui_components[element].layout, 'visibility', visibility) for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label'] if element in ui_components]

def reset_progress_component(ui_components: Dict[str, Any]) -> None:
    """Reset komponen progress dengan one-liner."""
    show_progress_component(ui_components, False)
    [setattr(ui_components[element], 'value', 0) for element in ['progress_bar', 'current_progress'] if element in ui_components]
    [setattr(ui_components[element], 'value', "") for element in ['overall_label', 'step_label'] if element in ui_components]

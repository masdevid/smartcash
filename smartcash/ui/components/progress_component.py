"""
File: smartcash/ui/components/progress_component.py
Deskripsi: Komponen progress tracking yang dapat digunakan bersama untuk berbagai modul dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_progress_component(module_name: str = "processing") -> Dict[str, Any]:
    """
    Membuat komponen progress tracking yang konsisten untuk berbagai modul.
    
    Args:
        module_name: Nama modul untuk kustomisasi label
    
    Returns:
        Dictionary berisi komponen progress tracking
    """
    # Progress bar utama dengan styling standar
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Total:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    # Progress bar untuk step saat ini
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Step:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    # Labels untuk progress
    overall_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0'))
    step_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0'))
    
    # Container progress dengan styling yang konsisten
    progress_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
        progress_bar,
        overall_label,
        current_progress,
        step_label
    ])
    
    # Return dictionary berisi semua komponen
    return {
        'container': progress_container,
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'overall_label': overall_label,
        'step_label': step_label
    }

def show_progress_component(ui_components: Dict[str, Any], show: bool = True) -> None:
    """
    Menampilkan atau menyembunyikan komponen progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        show: Apakah perlu ditampilkan
    """
    visibility = 'visible' if show else 'hidden'
    
    # Tampilkan progress bar dan label
    for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
        if element in ui_components:
            ui_components[element].layout.visibility = visibility

def reset_progress_component(ui_components: Dict[str, Any]) -> None:
    """
    Reset komponen progress tracking ke kondisi awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Sembunyikan progress bar
    show_progress_component(ui_components, False)
    
    # Reset nilai progress
    for element in ['progress_bar', 'current_progress']:
        if element in ui_components:
            ui_components[element].value = 0
    
    # Reset label
    for element in ['overall_label', 'step_label']:
        if element in ui_components:
            ui_components[element].value = ""

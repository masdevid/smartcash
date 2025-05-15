"""
File: smartcash/ui/training_config/training_strategy/components/button_components.py
Deskripsi: Komponen tombol untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET

def create_training_strategy_button_components() -> Dict[str, Any]:
    """
    Membuat komponen tombol untuk strategi pelatihan.
    
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {}
    
    # Tombol save
    ui_components['save_button'] = widgets.Button(
        description=f"{ICONS.get('save', '💾')} Simpan",
        button_style='primary',
        tooltip='Simpan konfigurasi strategi pelatihan dan sinkronkan ke Google Drive',
        layout=widgets.Layout(width='auto')
    )
    
    # Tombol reset
    ui_components['reset_button'] = widgets.Button(
        description=f"{ICONS.get('reset', '🔄')} Reset",
        button_style='warning',
        tooltip='Reset konfigurasi strategi pelatihan ke default',
        layout=widgets.Layout(width='auto')
    )
    
    # Status output
    ui_components['status'] = widgets.Output(
        layout=OUTPUT_WIDGET
    )
    
    # Container untuk tombol
    ui_components['button_container'] = widgets.HBox([
        ui_components['save_button'],
        ui_components['reset_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        justify_content='flex-start',
        align_items='center',
        width='100%',
        margin='10px 0',
        gap='10px'
    ))
    
    return ui_components

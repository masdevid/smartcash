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
        description='Simpan',
        button_style='primary',
        icon=ICONS.get('save', '💾'),
        tooltip='Simpan konfigurasi strategi pelatihan dan sinkronkan ke Google Drive',
        layout=widgets.Layout(width='100px')
    )
    
    # Tombol reset
    ui_components['reset_button'] = widgets.Button(
        description='Reset',
        button_style='warning',
        icon=ICONS.get('reset', '🔄'),
        tooltip='Reset konfigurasi strategi pelatihan ke default',
        layout=widgets.Layout(width='100px')
    )
    
    # Tambahkan keterangan sinkronisasi otomatis
    ui_components['sync_info'] = widgets.HTML(
        value=f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('info', 'ℹ️')} Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.</div>"
    )
    
    # Buat panel untuk status (seperti backbone)
    ui_components['status_panel'] = widgets.Output(
        layout=widgets.Layout(width='100%', min_height='50px')
    )
    
    # Container untuk tombol
    ui_components['button_container'] = widgets.HBox([
        ui_components['save_button'],
        ui_components['reset_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row nowrap',
        justify_content='flex-end',
        align_items='center',
        gap='10px',
        width='auto',
        margin='10px 0px'
    ))
    
    # Untuk kompatibilitas dengan kode lama
    ui_components['status'] = ui_components['status_panel']
    
    return ui_components

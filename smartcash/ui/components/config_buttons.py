"""
File: smartcash/ui/components/config_buttons.py
Deskripsi: Komponen tombol yang digunakan pada berbagai UI konfigurasi
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_config_buttons() -> Dict[str, Any]:
    """
    Membuat tombol-tombol standar untuk UI konfigurasi.
    
    Returns:
        Dict berisi tombol-tombol konfigurasi
    """
    buttons = {}
    
    # Buat tombol aksi
    buttons['save_button'] = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='success',
        icon='save',
        tooltip='Simpan konfigurasi',
        layout=widgets.Layout(width='200px')
    )
    
    buttons['reset_button'] = widgets.Button(
        description='Reset',
        button_style='warning',
        icon='refresh',
        tooltip='Reset ke default',
        layout=widgets.Layout(width='100px')
    )
    
    # Container tombol
    buttons['container'] = widgets.HBox(
        [buttons['save_button'], buttons['reset_button']],
        layout=widgets.Layout(padding='10px')
    )
    
    return buttons

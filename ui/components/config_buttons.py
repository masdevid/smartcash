"""
File: smartcash/ui/components/config_buttons.py
Deskripsi: Komponen tombol konfigurasi dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_config_buttons(title: str = None) -> Dict[str, Any]:
    """Membuat tombol-tombol standar untuk UI konfigurasi dengan one-liner."""
    save_button = widgets.Button(description='Simpan Konfigurasi', button_style='success', icon='save',
                                tooltip='Simpan konfigurasi', layout=widgets.Layout(width='200px'))
    reset_button = widgets.Button(description='Reset', button_style='warning', icon='refresh',
                                 tooltip='Reset ke default', layout=widgets.Layout(width='100px'))
    return {'save_button': save_button, 'reset_button': reset_button, 
            'container': widgets.HBox([save_button, reset_button], layout=widgets.Layout(padding='10px'))}

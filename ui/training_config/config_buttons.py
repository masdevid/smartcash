"""
File: smartcash/ui/training_config/config_buttons.py
Deskripsi: Komponen tombol konfigurasi bersama untuk UI training
"""

import ipywidgets as widgets

def create_config_buttons(context="Konfigurasi"):
    """
    Buat komponen tombol konfigurasi bersama.
    
    Args:
        context: Konteks untuk tooltip
        
    Returns:
        Widget container dengan tombol save dan reset
    """
    # Tombol save
    save_button = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='primary',
        icon='save',
        layout=widgets.Layout(margin='0 10px 0 0'),
        tooltip=f'Simpan {context} ke file konfigurasi'
    )
    
    # Tombol reset
    reset_button = widgets.Button(
        description='Reset ke Default',
        button_style='warning',
        icon='refresh',
        layout=widgets.Layout(margin='0'),
        tooltip=f'Reset {context} ke nilai default'
    )
    
    # Container
    return widgets.HBox(
        [save_button, reset_button],
        layout=widgets.Layout(
            display='flex',
            justify_content='flex-start',
            margin='10px 0'
        )
    )
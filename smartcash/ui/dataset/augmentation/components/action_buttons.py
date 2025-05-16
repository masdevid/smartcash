"""
File: smartcash/ui/dataset/augmentation/components/action_buttons.py
Deskripsi: Komponen UI untuk tombol aksi augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_action_buttons() -> Dict[str, Any]:
    """
    Buat komponen UI untuk tombol aksi augmentasi dataset.
    
    Returns:
        Dictionary berisi tombol aksi
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Tombol augmentasi
    augment_button = widgets.Button(
        description='Mulai Augmentasi',
        icon='random',
        button_style='primary',
        tooltip='Mulai proses augmentasi dataset',
        layout=widgets.Layout(width='auto')
    )
    
    # Tombol stop
    stop_button = widgets.Button(
        description='Stop',
        icon='stop',
        button_style='danger',
        tooltip='Hentikan proses augmentasi',
        layout=widgets.Layout(width='auto', display='none')
    )
    
    # Tombol reset
    reset_button = widgets.Button(
        description='Reset',
        icon='refresh',
        button_style='warning',
        tooltip='Reset konfigurasi ke default',
        layout=widgets.Layout(width='auto')
    )
    
    # Tombol cleanup dengan deskripsi yang lebih jelas
    cleanup_button = widgets.Button(
        description='Hapus Hasil',
        icon='trash',
        button_style='danger',
        tooltip='Hapus file hasil augmentasi tanpa backup',
        layout=widgets.Layout(width='auto', display='none')
    )
    
    # Tombol save
    save_button = widgets.Button(
        description='Simpan Konfigurasi',
        icon='save',
        button_style='success',
        tooltip='Simpan konfigurasi augmentasi',
        layout=widgets.Layout(width='auto')
    )
    
    # Container untuk tombol
    button_container = widgets.HBox([
        augment_button, 
        stop_button,
        reset_button,
        cleanup_button,
        save_button
    ], layout=widgets.Layout(
        justify_content='flex-start',
        align_items='center',
        margin='10px 0'
    ))
    
    # Dictionary berisi tombol
    buttons = {
        'primary_button': augment_button,
        'augment_button': augment_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'cleanup_button': cleanup_button,
        'save_button': save_button,
        'container': button_container
    }
    
    return buttons

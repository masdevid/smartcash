"""
File: smartcash/ui/helpers/action_buttons.py
Deskripsi: Komponen tombol aksi standar untuk preprocessing dan augmentasi dengan tampilan seragam
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple, Callable
from smartcash.ui.utils.constants import COLORS, ICONS

def create_action_buttons(
    primary_label: str = "Run Process", 
    primary_icon: str = "play",
    secondary_buttons: List[Tuple[str, str, str]] = None, 
    cleanup_enabled: bool = True,
    layout: Optional[widgets.Layout] = None
) -> Dict[str, widgets.Button]:
    """
    Buat set tombol aksi standar dengan tampilan seragam.
    
    Args:
        primary_label: Label untuk tombol aksi utama
        primary_icon: Icon untuk tombol aksi utama
        secondary_buttons: List tuple (label, icon, button_style) untuk tombol sekunder
        cleanup_enabled: Apakah tombol cleanup ditampilkan
        layout: Layout optional untuk container
        
    Returns:
        Dictionary berisi semua button widget
    """
    # Tombol utama dengan gaya primary
    primary_button = widgets.Button(
        description=primary_label,
        button_style='primary',
        icon=primary_icon,
        tooltip=f"Mulai proses {primary_label.lower()}",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol stop dengan hidden default
    stop_button = widgets.Button(
        description='Stop',
        button_style='danger',
        icon='stop',
        tooltip="Hentikan proses yang sedang berjalan",
        layout=widgets.Layout(display='none', margin='5px')
    )
    
    # Tombol reset
    reset_button = widgets.Button(
        description='Reset',
        button_style='warning',
        icon='refresh',
        tooltip="Reset konfigurasi dan tampilan",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol save
    save_button = widgets.Button(
        description='Simpan Konfigurasi',
        button_style='success',
        icon='save',
        tooltip="Simpan konfigurasi saat ini",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol cleanup dengan hidden default
    cleanup_button = widgets.Button(
        description='Hapus Data',
        button_style='danger',
        icon='trash',
        tooltip="Hapus data hasil proses",
        layout=widgets.Layout(display='none' if not cleanup_enabled else 'inline-block', margin='5px')
    )
    
    # List untuk menyimpan semua tombol
    all_buttons = [primary_button, stop_button, reset_button, save_button]
    
    # Tambahkan tombol cleanup jika diaktifkan
    if cleanup_enabled:
        all_buttons.append(cleanup_button)
    
    # Tambahkan tombol sekunder jika ada
    secondary_widget_buttons = []
    if secondary_buttons:
        for label, icon, button_style in secondary_buttons:
            button = widgets.Button(
                description=label,
                button_style=button_style,
                icon=icon,
                tooltip=f"{label}",
                layout=widgets.Layout(margin='5px')
            )
            secondary_widget_buttons.append(button)
            all_buttons.append(button)
    
    # Default layout jika tidak ada yang diberikan
    if not layout:
        layout = widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='center',
            width='100%',
            margin='10px 0',
            gap='5px'
        )
    
    # Buat container untuk semua tombol
    button_container = widgets.HBox(all_buttons, layout=layout)
    
    # Kembalikan dictionary berisi semua tombol dan container
    buttons = {
        'primary_button': primary_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'save_button': save_button,
        'cleanup_button': cleanup_button,
        'container': button_container
    }
    
    # Tambahkan tombol sekunder ke dictionary
    if secondary_buttons:
        buttons['secondary_buttons'] = secondary_widget_buttons
    
    return buttons

def create_visualization_buttons(layout: Optional[widgets.Layout] = None) -> Dict[str, Any]:
    """
    Buat tombol visualisasi standar untuk tampilan hasil.
    
    Args:
        layout: Layout optional untuk container
        
    Returns:
        Dictionary berisi tombol visualisasi dan container
    """
    # Tombol visualisasi sampel
    visualize_button = widgets.Button(
        description='Tampilkan Sampel',
        button_style='info',
        icon='image',
        tooltip="Tampilkan sampel hasil",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol komparasi
    compare_button = widgets.Button(
        description='Bandingkan Data',
        button_style='info',
        icon='columns',
        tooltip="Bandingkan data asli dan hasil",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol distribusi
    distribution_button = widgets.Button(
        description='Distribusi Kelas',
        button_style='info',
        icon='bar-chart',
        tooltip="Tampilkan distribusi kelas dataset",
        layout=widgets.Layout(margin='5px')
    )
    
    # Default layout jika tidak ada yang diberikan
    if not layout:
        layout = widgets.Layout(
            display='none',  # Hidden by default
            flex_flow='row wrap',
            align_items='center',
            width='100%',
            margin='10px 0',
            gap='5px'
        )
    
    # Buat container untuk tombol visualisasi
    button_container = widgets.HBox(
        [visualize_button, compare_button, distribution_button],
        layout=layout
    )
    
    # Kembalikan dictionary berisi tombol dan container
    return {
        'visualize_button': visualize_button,
        'compare_button': compare_button,
        'distribution_button': distribution_button,
        'container': button_container
    }
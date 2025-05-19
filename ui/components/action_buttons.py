"""
File: smartcash/ui/components/action_buttons.py
Deskripsi: Komponen tombol aksi reusable dengan tampilan standar
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple, Optional

def create_action_buttons(
    primary_label: str = "Run Process", 
    primary_icon: str = "play",
    secondary_buttons: List[Tuple[str, str, str]] = None, 
    cleanup_enabled: bool = True,
    layout: Optional[widgets.Layout] = None
) -> Dict[str, widgets.Widget]:
    """
    Buat set tombol aksi standar dengan tampilan seragam.
    
    Args:
        primary_label: Label untuk tombol aksi utama
        primary_icon: Icon untuk tombol aksi utama
        secondary_buttons: List tuple (label, icon, button_style) untuk tombol sekunder
        cleanup_enabled: Flag untuk menampilkan tombol cleanup
        layout: Layout untuk container
        
    Returns:
        Dictionary berisi tombol dan container widgets
    """
    # Tombol utama dengan gaya primary
    primary_button = widgets.Button(
        description=primary_label,
        button_style='primary',
        icon=primary_icon,
        tooltip=f"Mulai proses {primary_label.lower()}",
        layout=widgets.Layout(margin='5px')
    )
    
    # Tombol stop (hidden by default)
    stop_button = widgets.Button(
        description='Stop',
        button_style='danger',
        icon='stop',
        tooltip="Hentikan proses yang sedang berjalan",
        layout=widgets.Layout(display='none', margin='5px')
    )
    
    # Tidak lagi menggunakan tombol reset dan save karena sudah ada di save_reset_buttons
    
    # Tombol cleanup (optional)
    cleanup_button = widgets.Button(
        description='Hapus Data',
        button_style='danger',
        icon='trash',
        tooltip="Hapus data hasil proses",
        layout=widgets.Layout(
            display='none' if not cleanup_enabled else 'inline-block',
            margin='5px'
        )
    )
    
    # List untuk menyimpan semua tombol
    buttons = [primary_button, stop_button]
    
    # Tambahkan cleanup jika enabled
    if cleanup_enabled:
        buttons.append(cleanup_button)
    
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
            buttons.append(button)
    
    # Default layout
    if not layout:
        layout = widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='center',
            width='100%',
            margin='10px 0',
            gap='5px'
        )
    
    # Container untuk tombol
    button_container = widgets.HBox(buttons, layout=layout)
    
    # Kembalikan dictionary dengan semua komponen
    buttons_dict = {
        'primary_button': primary_button,
        'stop_button': stop_button,
        'cleanup_button': cleanup_button,
        'container': button_container
    }
    
    # Tambahkan tombol sekunder ke dictionary
    if secondary_buttons:
        buttons_dict['secondary_buttons'] = secondary_widget_buttons
    
    return buttons_dict

def create_visualization_buttons(layout: Optional[widgets.Layout] = None, include_distribution: bool = True) -> Dict[str, Any]:
    """
    Buat tombol visualisasi standar untuk tampilan hasil.
    
    Args:
        layout: Layout untuk container
        
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
    
    # Tombol distribusi (opsional)
    distribution_button = None
    if include_distribution:
        distribution_button = widgets.Button(
            description='Distribusi Kelas',
            button_style='info',
            icon='bar-chart',
            tooltip="Tampilkan distribusi kelas dataset",
            layout=widgets.Layout(margin='5px')
        )
    
    # Default layout
    if not layout:
        layout = widgets.Layout(
            display='none',  # Hidden by default
            flex_flow='row wrap',
            align_items='center',
            width='100%',
            margin='10px 0',
            gap='5px'
        )
    
    # Container untuk tombol visualisasi
    buttons = [visualize_button, compare_button]
    if distribution_button:
        buttons.append(distribution_button)
        
    button_container = widgets.HBox(
        buttons,
        layout=layout
    )
    
    # Kembalikan dictionary berisi tombol dan container
    result = {
        'visualize_button': visualize_button,
        'compare_button': compare_button,
        'container': button_container
    }
    
    # Tambahkan distribution_button ke result jika ada
    if distribution_button:
        result['distribution_button'] = distribution_button
        
    return result
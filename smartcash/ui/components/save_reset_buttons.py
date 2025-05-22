"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Komponen shared untuk tombol simpan dan reset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable
from smartcash.ui.utils.constants import ICONS, COLORS

def create_save_reset_buttons(
    save_label: str = "Simpan",
    reset_label: str = "Reset",
    save_tooltip: str = "Simpan konfigurasi",
    reset_tooltip: str = "Reset konfigurasi ke default",
    save_handler: Optional[Callable] = None,
    reset_handler: Optional[Callable] = None,
    save_icon: str = "save",
    reset_icon: str = "reset",
    with_sync_info: bool = False,
    sync_message: str = "Konfigurasi akan otomatis disinkronkan saat disimpan atau direset.",
    button_width: str = "100px",
    container_width: str = "auto"
) -> Dict[str, Any]:
    """
    Buat komponen tombol simpan dan reset yang dapat digunakan di berbagai modul.
    
    Args:
        save_label: Label untuk tombol simpan
        reset_label: Label untuk tombol reset
        save_tooltip: Tooltip untuk tombol simpan
        reset_tooltip: Tooltip untuk tombol reset
        save_handler: Handler untuk tombol simpan
        reset_handler: Handler untuk tombol reset
        save_icon: Ikon untuk tombol simpan
        reset_icon: Ikon untuk tombol reset
        with_sync_info: Apakah perlu menambahkan keterangan sinkronisasi
        sync_message: Pesan sinkronisasi
        button_width: Lebar tombol
        container_width: Lebar container
        
    Returns:
        Dictionary berisi komponen tombol simpan dan reset
    """
    # Buat tombol Save
    save_button = widgets.Button(
        description=save_label,
        button_style='primary',
        icon=ICONS.get(save_icon, '💾'),
        tooltip=save_tooltip,
        layout=widgets.Layout(width=button_width)
    )
    
    # Buat tombol Reset
    reset_button = widgets.Button(
        description=reset_label,
        button_style='warning',
        icon=ICONS.get(reset_icon, '🔄'),
        tooltip=reset_tooltip,
        layout=widgets.Layout(width=button_width)
    )
    
    # Register handler jika ada
    if save_handler:
        save_button.on_click(save_handler)
    
    if reset_handler:
        reset_button.on_click(reset_handler)
    
    # Buat container untuk tombol
    button_container = widgets.HBox([
        save_button,
        reset_button
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row nowrap',
        justify_content='flex-end',
        align_items='center',
        gap='10px',
        width=container_width,
        margin='10px 0px'
    ))
    
    # Tambahkan keterangan sinkronisasi jika diperlukan
    sync_info = None
    if with_sync_info:
        sync_info = widgets.HTML(
            value=f"<div style='margin-top: 5px; font-style: italic; color: #666;'>{ICONS.get('info', 'ℹ️')} {sync_message}</div>"
        )
    
    # Buat container untuk tombol dan sync info
    container = None
    if sync_info:
        container = widgets.VBox([
            button_container,
            sync_info
        ], layout=widgets.Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='flex-end',
            align_items='right',
            gap='10px',
            width=container_width,
            margin='10px 0px'
        ))
    else:
        container = button_container
    
    return {
        'save_button': save_button,
        'reset_button': reset_button,
        'button_container': button_container,
        'sync_info': sync_info,
        'container': container
    }

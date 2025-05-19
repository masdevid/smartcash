"""
File: smartcash/ui/components/sync_info_message.py
Deskripsi: Komponen shared untuk pesan informasi sinkronisasi
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS, COLORS

def create_sync_info_message(
    message: str = "Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
    icon: str = "info",
    color: str = "#666",
    font_style: str = "italic",
    margin_top: str = "5px",
    width: str = "100%"
) -> Dict[str, Any]:
    """
    Buat komponen pesan informasi sinkronisasi yang dapat digunakan di berbagai modul.
    
    Args:
        message: Pesan informasi sinkronisasi
        icon: Ikon untuk pesan
        color: Warna teks pesan
        font_style: Gaya font pesan
        margin_top: Margin atas pesan
        width: Lebar komponen
        
    Returns:
        Dictionary berisi komponen pesan informasi sinkronisasi
    """
    # Tambahkan ikon jika tersedia
    icon_html = ""
    if icon and icon in ICONS:
        icon_html = f"{ICONS.get(icon, 'ℹ️')} "
    
    # Buat komponen HTML untuk pesan dengan text-align: right
    sync_info = widgets.HTML(
        value=f"<div style='margin-top: {margin_top}; font-style: {font_style}; color: {color}; text-align: right;'>{icon_html}{message}</div>",
        layout=widgets.Layout(width=width)
    )
    
    return {
        'sync_info': sync_info,
        'message': message,
        'container': sync_info  # Alias untuk konsistensi dengan komponen lain
    }

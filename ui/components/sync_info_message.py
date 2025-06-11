"""
File: smartcash/ui/components/sync_info_message.py
Deskripsi: Komponen shared untuk pesan informasi sinkronisasi dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS

def create_sync_info_message(message: str = "Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
                            icon: str = "info", color: str = "#666", font_style: str = "italic",
                            margin_top: str = "5px", width: str = "100%") -> Dict[str, Any]:
    """Buat komponen pesan informasi sinkronisasi dengan one-liner style."""
    icon_html = f"{ICONS.get(icon, 'ℹ️')} " if icon and icon in ICONS else ""
    sync_info = widgets.HTML(f"<div style='margin-top: {margin_top}; font-style: {font_style}; color: {color}; text-align: right;'>{icon_html}{message}</div>",
                            layout=widgets.Layout(width=width))
    return {'sync_info': sync_info, 'message': message, 'container': sync_info}

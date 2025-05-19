"""
File: smartcash/ui/components/info_accordion.py
Deskripsi: Komponen accordion untuk menampilkan informasi dengan pendekatan DRY
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS

def create_info_accordion(
    title: str = "Informasi",
    content: Any = None,
    icon: str = "info",
    open_by_default: bool = False
) -> Dict[str, Any]:
    """
    Membuat accordion untuk menampilkan informasi dengan pendekatan DRY.
    
    Args:
        title: Judul accordion
        content: Konten accordion (widget atau HTML)
        icon: Ikon untuk judul accordion
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Dictionary berisi komponen accordion
    """
    # Gunakan ikon dari constants jika tersedia
    icon_display = ICONS.get(icon, "ℹ️")
    
    # Buat container untuk konten
    if content is None:
        content = widgets.HTML(
            f"<div style='padding: 10px; background-color: #f8f9fa;'>"
            f"<p>Tidak ada informasi yang tersedia.</p>"
            f"</div>"
        )
    
    # Buat accordion
    accordion = widgets.Accordion(children=[content])
    accordion.set_title(0, f"{icon_display} {title}")
    
    # Set selected index
    accordion.selected_index = 0 if open_by_default else None
    
    # Buat container untuk accordion
    container = widgets.VBox([
        accordion
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        padding='0'
    ))
    
    # Kembalikan dictionary berisi komponen
    return {
        'accordion': accordion,
        'container': container,
        'content': content,
        'title': title
    }

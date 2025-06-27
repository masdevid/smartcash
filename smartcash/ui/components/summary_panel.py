"""
File: smartcash/ui/components/summary_panel.py
Deskripsi: Komponen untuk menampilkan ringkasan informasi dalam panel
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional, List, Callable

def create_summary_panel(
    title: str = "Summary",
    items: Optional[Dict[str, Any]] = None,
    layout: Optional[Dict[str, Any]] = None,
    **kwargs
) -> widgets.VBox:
    """Buat panel ringkasan dengan daftar item
    
    Args:
        title: Judul panel
        items: Dictionary berisi item-item ringkasan
        layout: Layout kustom untuk panel
        **kwargs: Argumen tambahan untuk panel
        
    Returns:
        widgets.VBox: Panel ringkasan
    """
    # Default layout
    default_layout = {
        'width': '100%',
        'margin': '10px 0',
        'padding': '10px',
        'border': '1px solid #e0e0e0',
        'border_radius': '5px',
        'background_color': '#f9f9f9',
        **kwargs
    }
    
    # Gabungkan dengan layout kustom jika ada
    if layout:
        default_layout.update(layout)
    
    # Buat header
    header = widgets.HTML(
        f"<h4 style='margin: 0 0 10px 0;'>{title}</h4>"
    )
    
    # Buat konten
    content = widgets.VBox(layout={'margin': '0 0 0 10px'})
    
    # Tambahkan item jika ada
    if items:
        item_widgets = []
        for key, value in items.items():
            item = widgets.HTML(
                f"<div style='margin: 5px 0;'><b>{key}:</b> {value}</div>"
            )
            item_widgets.append(item)
        content.children = item_widgets
    
    # Buat panel
    panel = widgets.VBox(
        [header, content],
        layout=widgets.Layout(**default_layout)
    )
    
    return panel

def update_summary_panel(
    panel: widgets.VBox,
    items: Dict[str, Any],
    title: Optional[str] = None
) -> None:
    """Update isi panel ringkasan
    
    Args:
        panel: Widget panel yang akan diupdate
        items: Dictionary berisi item-item ringkasan
        title: Judul baru (opsional)
    """
    if not isinstance(panel, widgets.VBox) or len(panel.children) < 2:
        return
    
    # Update judul jika disediakan
    if title is not None and isinstance(panel.children[0], widgets.HTML):
        panel.children[0].value = f"<h4 style='margin: 0 0 10px 0;'>{title}</h4>"
    
    # Update konten
    content = panel.children[1]
    if not isinstance(content, widgets.VBox):
        return
    
    item_widgets = []
    for key, value in items.items():
        item = widgets.HTML(
            f"<div style='margin: 5px 0;'><b>{key}:</b> {value}</div>"
        )
        item_widgets.append(item)
    
    content.children = item_widgets

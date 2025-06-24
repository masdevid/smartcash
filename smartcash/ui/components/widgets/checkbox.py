"""
File: smartcash/ui/components/widgets/checkbox.py
Deskripsi: Komponen checkbox yang dapat digunakan kembali
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.layout.layout_components import create_element

def create_checkbox(
    name: str,
    value: bool,
    description: str,
    tooltip: str = "",
    **kwargs
) -> widgets.Checkbox:
    """
    Membuat checkbox dengan konfigurasi yang responsif
    
    Args:
        name: Nama unik untuk checkbox
        value: Nilai awal (True/False)
        description: Deskripsi yang ditampilkan di sebelah checkbox
        tooltip: Tooltip yang muncul saat hover
        **kwargs: Argumen tambahan untuk konfigurasi
        
    Returns:
        Instance widgets.Checkbox yang sudah dikonfigurasi
    """
    return create_element(
        'checkbox', 
        value=value, 
        description=description, 
        tooltip=tooltip,
        **kwargs
    )

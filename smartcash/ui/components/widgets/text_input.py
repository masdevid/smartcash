"""
File: smartcash/ui/components/widgets/text_input.py
Deskripsi: Komponen text input yang dapat digunakan kembali
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.layout.layout_components import create_element

def create_text_input(
    name: str,
    value: str,
    description: str,
    placeholder: str = "",
    tooltip: str = "",
    **kwargs
) -> widgets.Text:
    """
    Membuat text input dengan konfigurasi yang responsif
    
    Args:
        name: Nama unik untuk text input
        value: Nilai awal
        description: Label yang ditampilkan di sebelah input
        placeholder: Teks placeholder
        tooltip: Tooltip yang muncul saat hover
        **kwargs: Argumen tambahan untuk konfigurasi
        
    Returns:
        Instance widgets.Text yang sudah dikonfigurasi
    """
    return create_element(
        'text_input', 
        value=value, 
        description=description, 
        placeholder=placeholder, 
        tooltip=tooltip,
        **kwargs
    )

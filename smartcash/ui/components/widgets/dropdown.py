"""
File: smartcash/ui/components/widgets/dropdown.py
Deskripsi: Komponen dropdown yang dapat digunakan kembali
"""

from typing import Any, Dict, List, Optional, Union
import ipywidgets as widgets
from smartcash.ui.components.layout.layout_components import create_element

def create_dropdown(
    name: str,
    value: Any,
    options: Union[List[Any], Dict[str, Any]],
    description: str,
    tooltip: str = "",
    **kwargs
) -> widgets.Dropdown:
    """
    Membuat dropdown dengan konfigurasi yang responsif
    
    Args:
        name: Nama unik untuk dropdown
        value: Nilai yang dipilih
        options: Daftar opsi atau dictionary {value: label}
        description: Deskripsi yang ditampilkan di sebelah dropdown
        tooltip: Tooltip yang muncul saat hover
        **kwargs: Argumen tambahan untuk konfigurasi
        
    Returns:
        Instance widgets.Dropdown yang sudah dikonfigurasi
    """
    return create_element(
        'dropdown', 
        value=value, 
        options=options, 
        description=description, 
        tooltip=tooltip,
        **kwargs
    )

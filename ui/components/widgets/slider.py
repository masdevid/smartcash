"""
File: smartcash/ui/components/widgets/slider.py
Deskripsi: Komponen slider yang dapat digunakan kembali
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.layout.layout_components import create_element

def create_slider(
    name: str,
    value: float,
    min_val: float,
    max_val: float,
    step: float,
    description: str,
    tooltip: str = "",
    style: Optional[Dict[str, Any]] = None
) -> widgets.FloatSlider:
    """
    Membuat slider dengan konfigurasi yang responsif
    
    Args:
        name: Nama unik untuk slider
        value: Nilai awal
        min_val: Nilai minimum
        max_val: Nilai maksimum
        step: Langkah kenaikan/penurunan nilai
        description: Deskripsi yang ditampilkan di sebelah slider
        tooltip: Tooltip yang muncul saat hover
        style: Gaya tambahan untuk slider
        
    Returns:
        Instance widgets.FloatSlider yang sudah dikonfigurasi
    """
    return create_element(
        'slider', 
        value=value, 
        min_val=min_val, 
        max_val=max_val, 
        step=step, 
        description=description, 
        tooltip=tooltip, 
        style=style or {'description_width': '140px', 'handle_color': '#4CAF50'}
    )

"""
File: smartcash/ui/components/widgets/log_slider.py
Deskripsi: Komponen log slider yang dapat digunakan kembali
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.layout.layout_components import create_element

def create_log_slider(
    name: str,
    value: float,
    min_val: float,
    max_val: float,
    step: float,
    description: str,
    tooltip: str = "",
    **kwargs
) -> widgets.FloatSlider:
    """
    Membuat log slider dengan konfigurasi yang responsif
    
    Args:
        name: Nama unik untuk slider
        value: Nilai awal
        min_val: Nilai minimum
        max_val: Nilai maksimum
        step: Langkah kenaikan/penurunan nilai (dalam skala logaritmik)
        description: Deskripsi yang ditampilkan di sebelah slider
        tooltip: Tooltip yang muncul saat hover
        **kwargs: Argumen tambahan untuk konfigurasi
        
    Returns:
        Instance widgets.FloatSlider yang sudah dikonfigurasi dengan skala logaritmik
    """
    return create_element(
        'log_slider', 
        value=value, 
        min_val=min_val, 
        max_val=max_val, 
        step=step, 
        description=description, 
        tooltip=tooltip,
        **kwargs
    )

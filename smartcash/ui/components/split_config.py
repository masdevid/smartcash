"""
File: smartcash/ui/components/split_config.py
Deskripsi: Komponen shared untuk konfigurasi split dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from smartcash.ui.utils.constants import ICONS, COLORS

def create_split_config(
    title: str = "Konfigurasi Split Dataset",
    description: str = "Tentukan pembagian dataset untuk training, validation, dan testing",
    train_value: float = 0.7,
    val_value: float = 0.2,
    test_value: float = 0.1,
    min_value: float = 0.0,
    max_value: float = 1.0,
    step: float = 0.05,
    width: str = "100%",
    icon: str = "split"
) -> Dict[str, Any]:
    """
    Buat komponen konfigurasi split dataset yang dapat digunakan di berbagai modul.
    
    Args:
        title: Judul konfigurasi
        description: Deskripsi konfigurasi
        train_value: Nilai default untuk training split
        val_value: Nilai default untuk validation split
        test_value: Nilai default untuk testing split
        min_value: Nilai minimum untuk slider
        max_value: Nilai maksimum untuk slider
        step: Langkah untuk slider
        width: Lebar komponen
        icon: Ikon untuk judul
        
    Returns:
        Dictionary berisi komponen konfigurasi split
    """
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat header untuk konfigurasi
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat deskripsi jika ada
    description_widget = None
    if description:
        description_widget = widgets.HTML(
            value=f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>"
        )
    
    # Buat slider untuk training split
    train_slider = widgets.FloatSlider(
        value=train_value,
        min=min_value,
        max=max_value,
        step=step,
        description='Training:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width=width),
        readout_format='.0%'
    )
    
    # Buat slider untuk validation split
    val_slider = widgets.FloatSlider(
        value=val_value,
        min=min_value,
        max=max_value,
        step=step,
        description='Validation:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width=width),
        readout_format='.0%'
    )
    
    # Buat slider untuk testing split
    test_slider = widgets.FloatSlider(
        value=test_value,
        min=min_value,
        max=max_value,
        step=step,
        description='Testing:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width=width),
        readout_format='.0%'
    )
    
    # Buat output untuk menampilkan total
    total_output = widgets.HTML(
        value=f"<div style='margin-top: 10px;'><b>Total:</b> {(train_value + val_value + test_value) * 100:.0f}%</div>"
    )
    
    # Fungsi untuk memperbarui total
    def update_total(*args):
        total = train_slider.value + val_slider.value + test_slider.value
        color = COLORS.get('success', 'green') if abs(total - 1.0) < 0.01 else COLORS.get('error', 'red')
        total_output.value = f"<div style='margin-top: 10px;'><b>Total:</b> <span style='color: {color};'>{total * 100:.0f}%</span></div>"
    
    # Tambahkan observer untuk slider
    train_slider.observe(update_total, names='value')
    val_slider.observe(update_total, names='value')
    test_slider.observe(update_total, names='value')
    
    # Buat container untuk konfigurasi
    widgets_list = [header]
    if description_widget:
        widgets_list.append(description_widget)
    widgets_list.extend([train_slider, val_slider, test_slider, total_output])
    
    container = widgets.VBox(
        widgets_list,
        layout=widgets.Layout(
            margin='10px 0px',
            padding='10px',
            border='1px solid #eee',
            border_radius='4px'
        )
    )
    
    return {
        'container': container,
        'train_slider': train_slider,
        'val_slider': val_slider,
        'test_slider': test_slider,
        'total_output': total_output,
        'header': header
    }

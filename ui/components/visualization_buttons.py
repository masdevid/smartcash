"""
File: smartcash/ui/components/visualization_buttons.py
Deskripsi: Komponen tombol visualisasi yang dapat digunakan bersama untuk berbagai modul dataset
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS

def create_visualization_buttons(module_name: str = "dataset") -> Dict[str, Any]:
    """
    Membuat tombol-tombol visualisasi yang konsisten untuk berbagai modul.
    
    Args:
        module_name: Nama modul untuk kustomisasi label tombol
    
    Returns:
        Dictionary berisi tombol-tombol visualisasi
    """
    # Style tombol yang konsisten
    button_style = {
        'button_color': COLORS['primary'],
        'text_color': 'white',
        'border_radius': '4px',
        'margin': '5px'
    }
    
    # Tombol visualisasi sampel
    visualize_button = widgets.Button(
        description=f"{ICONS['chart']} Sampel",
        tooltip=f"Visualisasi sampel {module_name}",
        button_style='info',
        layout=widgets.Layout(width='auto')
    )
    
    # Tombol komparasi
    compare_button = widgets.Button(
        description=f"{ICONS['compare']} Komparasi",
        tooltip=f"Komparasi original vs {module_name}",
        button_style='info',
        layout=widgets.Layout(width='auto')
    )
    
    # Tombol distribusi
    distribution_button = widgets.Button(
        description=f"{ICONS['stats']} Distribusi",
        tooltip="Visualisasi distribusi kelas",
        button_style='info',
        layout=widgets.Layout(width='auto')
    )
    
    # Styling tombol dengan CSS
    for btn in [visualize_button, compare_button, distribution_button]:
        btn._dom_classes = ('custom-button',)
    
    # Container untuk tombol-tombol
    button_container = widgets.HBox(
        [visualize_button, compare_button, distribution_button],
        layout=widgets.Layout(
            display='none',
            margin='10px 0',
            justify_content='flex-start'
        )
    )
    
    # Tambahkan CSS styling
    style_html = f"""
    <style>
        .custom-button {{
            background-color: {button_style['button_color']} !important;
            color: {button_style['text_color']} !important;
            border-radius: {button_style['border_radius']} !important;
            margin: {button_style['margin']} !important;
        }}
    </style>
    """
    
    style_widget = widgets.HTML(value=style_html)
    
    # Gabungkan style dan container
    container_with_style = widgets.VBox([
        style_widget,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['chart']} Visualisasi</h4>"),
        button_container
    ])
    
    # Return dictionary berisi semua komponen
    return {
        'container': container_with_style,
        'button_container': button_container,
        'visualize_button': visualize_button,
        'compare_button': compare_button,
        'distribution_button': distribution_button,
        'style_widget': style_widget
    }

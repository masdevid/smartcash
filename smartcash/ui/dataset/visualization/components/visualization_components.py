"""
File: smartcash/ui/dataset/visualization/components/visualization_components.py
Deskripsi: Komponen UI untuk visualisasi dataset dengan pendekatan DRY
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.tab_factory import create_tab_widget
from smartcash.ui.components.accordion_factory import create_accordion

def create_visualization_components() -> Dict[str, Any]:
    """
    Membuat komponen UI untuk visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI yang telah dibuat
    """
    # Container utama
    main_container = widgets.VBox([])
    
    # Status panel
    status = widgets.Output()
    
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width='100%', visibility='hidden')
    )
    
    # Tab untuk visualisasi dataset
    tab_items = []
    
    # Tab 1: Distribusi Kelas
    class_distribution_container = widgets.VBox([])
    class_distribution_output = widgets.Output()
    class_distribution_button = widgets.Button(
        description=f"{ICONS.get('chart', '📊')} Tampilkan Distribusi Kelas",
        button_style='info',
        tooltip='Tampilkan distribusi kelas dataset'
    )
    class_distribution_container.children = [class_distribution_button, class_distribution_output]
    tab_items.append(('Distribusi Kelas', class_distribution_container))
    
    # Tab 2: Sampel Gambar
    sample_images_container = widgets.VBox([])
    sample_images_output = widgets.Output()
    sample_images_button = widgets.Button(
        description=f"{ICONS.get('image', '🖼️')} Tampilkan Sampel Gambar",
        button_style='info',
        tooltip='Tampilkan sampel gambar dari dataset'
    )
    sample_images_container.children = [sample_images_button, sample_images_output]
    tab_items.append(('Sampel Gambar', sample_images_container))
    
    # Tab 3: Statistik Dataset
    stats_container = widgets.VBox([])
    stats_output = widgets.Output()
    stats_button = widgets.Button(
        description=f"{ICONS.get('stats', '📈')} Tampilkan Statistik Dataset",
        button_style='info',
        tooltip='Tampilkan statistik dataset'
    )
    stats_container.children = [stats_button, stats_output]
    tab_items.append(('Statistik Dataset', stats_container))
    
    # Buat tab widget
    tab = create_tab_widget(tab_items)
    
    # Tambahkan komponen ke container utama
    main_container.children = [
        widgets.HTML("<h2>🔍 Visualisasi Dataset</h2>"),
        widgets.HTML("<p>Visualisasi dan analisis dataset untuk deteksi mata uang</p>"),
        status,
        progress_bar,
        tab
    ]
    
    # Kumpulkan semua komponen dalam dictionary
    ui_components = {
        'main_container': main_container,
        'status': status,
        'progress_bar': progress_bar,
        'tab': tab,
        'class_distribution_button': class_distribution_button,
        'class_distribution_output': class_distribution_output,
        'sample_images_button': sample_images_button,
        'sample_images_output': sample_images_output,
        'stats_button': stats_button,
        'stats_output': stats_output
    }
    
    return ui_components

"""
File: smartcash/ui/dataset/preprocessing/components/advanced_options.py
Deskripsi: Komponen advanced options untuk preprocessing
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.constants import COLORS, ICONS


def create_preprocessing_advanced_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen advanced options untuk preprocessing.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        widgets.VBox: Container advanced options
    """
    # Default config
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    
    # Worker slider
    num_workers = preprocessing_config.get('num_workers', 1)  # Colab safe default
    worker_slider = widgets.IntSlider(
        value=num_workers,
        min=1,
        max=4,  # Colab limit
        step=1,
        description='Workers:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    # Split selector
    split = preprocessing_config.get('split', 'all')
    split_map = {'all': 'Semua Split', 'train': 'Training', 'val': 'Validasi', 'test': 'Testing'}
    reverse_split_map = {v: k for k, v in split_map.items()}
    
    split_options = list(split_map.values())
    current_split = split_map.get(split, 'Semua Split')
    
    split_selector = widgets.Dropdown(
        options=split_options,
        value=current_split,
        description='Target Split:',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    # Container
    advanced_container = widgets.VBox([
        widgets.HTML(f"<h5 style='margin-bottom: 10px; color: {COLORS['dark']};'>{ICONS['config']} Opsi Lanjutan</h5>"),
        widgets.HTML("<div style='margin-bottom: 8px;'><b>Worker Thread:</b> Jumlah thread untuk preprocessing</div>"),
        worker_slider,
        widgets.HTML("<div style='margin: 8px 0;'><b>Target Split:</b> Bagian dataset yang akan diproses</div>"),
        split_selector
    ], layout=widgets.Layout(padding='10px 5px', width='48%'))
    
    # Tambahkan referensi untuk akses mudah
    advanced_container.worker_slider = worker_slider
    advanced_container.split_selector = split_selector
    advanced_container.reverse_split_map = reverse_split_map
    
    return advanced_container
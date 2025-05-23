"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Komponen input options yang disederhanakan untuk preprocessing dengan parameter sesuai backend
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS


def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen input options yang disederhanakan untuk preprocessing.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        widgets.VBox: Container input options
    """
    # Default config
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    
    # Extract nilai dari config dengan fallback yang sesuai backend
    img_size = preprocessing_config.get('img_size', (640, 640))
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        resolution_str = f"{img_size[0]}x{img_size[1]}"
    else:
        resolution_str = "640x640"
    
    # Normalization sesuai dengan backend
    normalization = preprocessing_config.get('normalization', 'minmax')
    
    # Resolution dropdown (maksimal 640x640 untuk efisiensi)
    resolution_dropdown = widgets.Dropdown(
        options=['320x320', '416x416', '512x512', '640x640'],
        value=resolution_str if resolution_str in ['320x320', '416x416', '512x512', '640x640'] else '640x640',
        description='Resolusi:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    # Normalization dropdown sesuai backend preprocessing
    normalization_options = ['minmax', 'standard', 'none']
    normalization_dropdown = widgets.Dropdown(
        options=normalization_options,
        value=normalization if normalization in normalization_options else 'minmax',
        description='Normalisasi:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    # Worker slider (default 4, max 10 untuk Colab)
    num_workers = preprocessing_config.get('num_workers', 4)
    worker_slider = widgets.IntSlider(
        value=min(max(num_workers, 1), 10),
        min=1,
        max=10,
        step=1,
        description='Workers:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    # Split selector
    split = preprocessing_config.get('split', 'all')
    split_options = ['all', 'train', 'val', 'test']
    split_dropdown = widgets.Dropdown(
        options=split_options,
        value=split if split in split_options else 'all',
        description='Target Split:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='95%', margin='5px 0')
    )
    
    # Container dengan header yang jelas
    options_container = widgets.VBox([
        widgets.HTML(f"<h5 style='margin-bottom: 10px; color: {COLORS['dark']};'>{ICONS['settings']} Opsi Preprocessing</h5>"),
        widgets.HTML("<div style='margin-bottom: 8px; color: #666;'><b>Resolusi Gambar:</b> Ukuran output preprocessing</div>"),
        resolution_dropdown,
        widgets.HTML("<div style='margin: 8px 0; color: #666;'><b>Normalisasi Data:</b> Metode normalisasi pixel</div>"),
        normalization_dropdown,
        widgets.HTML("<div style='margin: 8px 0; color: #666;'><b>Worker Threads:</b> Jumlah thread untuk preprocessing paralel</div>"),
        worker_slider,
        widgets.HTML("<div style='margin: 8px 0; color: #666;'><b>Target Split:</b> Bagian dataset yang akan diproses</div>"),
        split_dropdown
    ], layout=widgets.Layout(padding='15px', width='100%'))
    
    # Tambahkan referensi untuk akses mudah
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.worker_slider = worker_slider
    options_container.split_dropdown = split_dropdown
    
    return options_container
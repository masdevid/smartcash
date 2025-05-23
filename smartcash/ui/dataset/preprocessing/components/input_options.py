"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Komponen input options dalam layout 2 kolom untuk preprocessing dengan parameter sesuai backend
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS


def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen input options dalam layout 2 kolom untuk preprocessing.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        widgets.VBox: Container input options dengan layout 2 kolom
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
    
    # === KOLOM KIRI: Resolusi & Normalisasi ===
    
    # Resolution dropdown
    resolution_dropdown = widgets.Dropdown(
        options=['320x320', '416x416', '512x512', '640x640'],
        value=resolution_str if resolution_str in ['320x320', '416x416', '512x512', '640x640'] else '640x640',
        description='Resolusi:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Normalization dropdown
    normalization_options = ['minmax', 'standard', 'none']
    normalization_dropdown = widgets.Dropdown(
        options=normalization_options,
        value=normalization if normalization in normalization_options else 'minmax',
        description='Normalisasi:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    left_column = widgets.VBox([
        widgets.HTML("<div style='margin-bottom: 8px; color: #666; font-weight: bold;'>🖼️ Data & Format</div>"),
        widgets.HTML("<div style='margin-bottom: 5px; color: #888; font-size: 12px;'>Ukuran output preprocessing</div>"),
        resolution_dropdown,
        widgets.HTML("<div style='margin: 8px 0 5px 0; color: #888; font-size: 12px;'>Metode normalisasi pixel</div>"),
        normalization_dropdown
    ], layout=widgets.Layout(width='48%', padding='10px'))
    
    # === KOLOM KANAN: Worker & Split ===
    
    # Worker slider
    num_workers = preprocessing_config.get('num_workers', 4)
    worker_slider = widgets.IntSlider(
        value=min(max(num_workers, 1), 10),
        min=1,
        max=10,
        step=1,
        description='Workers:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Split selector
    split = preprocessing_config.get('split', 'all')
    split_options = ['all', 'train', 'val', 'test']
    split_dropdown = widgets.Dropdown(
        options=split_options,
        value=split if split in split_options else 'all',
        description='Target Split:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    right_column = widgets.VBox([
        widgets.HTML("<div style='margin-bottom: 8px; color: #666; font-weight: bold;'>⚙️ Performance & Scope</div>"),
        widgets.HTML("<div style='margin-bottom: 5px; color: #888; font-size: 12px;'>Jumlah thread paralel</div>"),
        worker_slider,
        widgets.HTML("<div style='margin: 8px 0 5px 0; color: #888; font-size: 12px;'>Bagian dataset yang diproses</div>"),
        split_dropdown
    ], layout=widgets.Layout(width='48%', padding='10px'))
    
    # Container 2 kolom
    columns_container = widgets.HBox([
        left_column, 
        right_column
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='flex-start'
    ))
    
    # Container utama dengan header
    options_container = widgets.VBox([
        widgets.HTML(f"<h5 style='margin-bottom: 15px; color: {COLORS['dark']};'>{ICONS['settings']} Opsi Preprocessing</h5>"),
        columns_container
    ], layout=widgets.Layout(
        padding='15px', 
        width='100%',
        border='1px solid #dee2e6',
        border_radius='5px',
        background_color='#f8f9fa'
    ))
    
    # Tambahkan referensi untuk akses mudah
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.worker_slider = worker_slider
    options_container.split_dropdown = split_dropdown
    
    return options_container
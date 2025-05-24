"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Fixed input options dengan responsive layout dan no horizontal scroll
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS


def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen input options dengan responsive layout dan no horizontal scroll.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        widgets.VBox: Container input options dengan fixed responsive layout
    """
    # Default config dengan safe extraction
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    
    # Extract nilai dari config dengan improved normalization handling
    img_size = preprocessing_config.get('img_size', (640, 640))
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        resolution_str = f"{img_size[0]}x{img_size[1]}"
    else:
        resolution_str = "640x640"
    
    # Improved normalization extraction
    normalization_raw = preprocessing_config.get('normalization', 'minmax')
    if isinstance(normalization_raw, dict):
        # Extract dari dict config
        enabled = normalization_raw.get('enabled', True)
        method = normalization_raw.get('method', 'minmax')
        
        if not enabled:
            normalization = 'none'
        else:
            # Map method names
            method_mapping = {
                'minmax': 'minmax',
                'min-max': 'minmax',
                'zscore': 'standard', 
                'z-score': 'standard',
                'standardization': 'standard',
                'whitening': 'standard'
            }
            normalization = method_mapping.get(method.lower(), 'minmax')
    else:
        normalization = normalization_raw if normalization_raw in ['minmax', 'standard', 'none'] else 'minmax'
    
    # === KOLOM KIRI: Resolusi & Normalisasi (Fixed Width) ===
    
    # Resolution dropdown dengan consistent width
    resolution_dropdown = widgets.Dropdown(
        options=['320x320', '416x416', '512x512', '640x640'],
        value=resolution_str if resolution_str in ['320x320', '416x416', '512x512', '640x640'] else '640x640',
        description='Resolusi:',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Normalization dropdown dengan consistent width
    normalization_options = ['minmax', 'standard', 'none']
    normalization_dropdown = widgets.Dropdown(
        options=normalization_options,
        value=normalization,
        description='Normalisasi:',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Left column dengan fixed padding
    left_column = widgets.VBox([
        widgets.HTML("<div style='margin-bottom: 6px; color: #666; font-weight: bold; font-size: 13px;'>🖼️ Data & Format</div>"),
        widgets.HTML("<div style='margin-bottom: 3px; color: #888; font-size: 11px;'>Ukuran output preprocessing</div>"),
        resolution_dropdown,
        widgets.HTML("<div style='margin: 6px 0 3px 0; color: #888; font-size: 11px;'>Metode normalisasi pixel</div>"),
        normalization_dropdown
    ], layout=widgets.Layout(width='47%', padding='8px'))
    
    # === KOLOM KANAN: Worker & Split (Fixed Width) ===
    
    # Worker slider dengan consistent width
    num_workers = preprocessing_config.get('num_workers', 4)
    worker_slider = widgets.IntSlider(
        value=min(max(num_workers, 1), 10),
        min=1,
        max=10,
        step=1,
        description='Workers:',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Split selector dengan consistent width
    split = preprocessing_config.get('split', 'all')
    split_options = ['all', 'train', 'valid', 'test']  # Use 'valid' instead of 'val'
    split_dropdown = widgets.Dropdown(
        options=split_options,
        value=split if split in split_options else 'all',
        description='Target Split:',
        style={'description_width': '70px'},
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Right column dengan fixed padding
    right_column = widgets.VBox([
        widgets.HTML("<div style='margin-bottom: 6px; color: #666; font-weight: bold; font-size: 13px;'>⚙️ Performance & Scope</div>"),
        widgets.HTML("<div style='margin-bottom: 3px; color: #888; font-size: 11px;'>Jumlah thread paralel</div>"),
        worker_slider,
        widgets.HTML("<div style='margin: 6px 0 3px 0; color: #888; font-size: 11px;'>Bagian dataset yang diproses</div>"),
        split_dropdown
    ], layout=widgets.Layout(width='47%', padding='8px'))
    
    # Container 2 kolom dengan responsive spacing (NO HORIZONTAL SCROLL)
    columns_container = widgets.HBox([
        left_column, 
        right_column
    ], layout=widgets.Layout(
        width='100%',
        justify_content='space-between',
        align_items='flex-start',
        margin='0px',  # Remove margin untuk prevent overflow
        padding='0px'  # Remove padding untuk prevent overflow
    ))
    
    # Safe icon dan color access
    def get_safe_icon(key: str, fallback: str = "⚙️") -> str:
        try:
            return ICONS.get(key, fallback)
        except (NameError, AttributeError):
            return fallback
    
    def get_safe_color(key: str, fallback: str = "#333") -> str:
        try:
            return COLORS.get(key, fallback)
        except (NameError, AttributeError):
            return fallback
    
    # Container utama dengan responsive design (NO OVERFLOW)
    options_container = widgets.VBox([
        widgets.HTML(f"<h5 style='margin: 10px 0 8px 0; color: {get_safe_color('dark', '#333')};'>{get_safe_icon('settings', '⚙️')} Opsi Preprocessing</h5>"),
        columns_container
    ], layout=widgets.Layout(
        padding='12px',
        width='100%',
        max_width='100%',  # Prevent overflow
        border='1px solid #dee2e6',
        border_radius='5px',
        background_color='#f8f9fa',
        overflow='hidden'  # Prevent any overflow
    ))
    
    # Tambahkan referensi untuk akses mudah
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.worker_slider = worker_slider
    options_container.split_dropdown = split_dropdown
    
    return options_container
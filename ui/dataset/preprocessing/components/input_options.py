"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Komponen opsi input untuk preprocessing dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE

def create_preprocessing_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi preprocessing.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi preprocessing
    """
    # Dapatkan nilai default dari config jika tersedia
    img_size = DEFAULT_IMG_SIZE[0]  # Default 640
    enable_normalization = True
    preserve_aspect_ratio = True
    enable_cache = True
    num_workers = 4
    
    if config and 'preprocessing' in config:
        preproc_config = config['preprocessing']
        # Ekstrak image size (array atau integer)
        if 'img_size' in preproc_config:
            img_size_val = preproc_config['img_size']
            img_size = img_size_val[0] if isinstance(img_size_val, (list, tuple)) else img_size_val
        
        # Ekstrak opsi normalisasi
        if 'normalization' in preproc_config:
            norm_config = preproc_config['normalization']
            enable_normalization = norm_config.get('enabled', True)
            preserve_aspect_ratio = norm_config.get('preserve_aspect_ratio', True)
        
        # Ekstrak pengaturan cache dan workers
        enable_cache = preproc_config.get('enabled', True)
        num_workers = preproc_config.get('num_workers', 4)
    
    # Buat komponen-komponen UI
    img_size_slider = widgets.IntSlider(
        value=img_size,
        min=320,
        max=960,
        step=32,
        description='Image size:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    enable_norm_checkbox = widgets.Checkbox(
        value=enable_normalization,
        description='Enable normalization',
        style={'description_width': 'initial'}
    )
    
    preserve_ratio_checkbox = widgets.Checkbox(
        value=preserve_aspect_ratio,
        description='Preserve aspect ratio',
        style={'description_width': 'initial'}
    )
    
    enable_cache_checkbox = widgets.Checkbox(
        value=enable_cache,
        description='Enable caching',
        style={'description_width': 'initial'}
    )
    
    num_workers_slider = widgets.IntSlider(
        value=num_workers,
        min=1,
        max=16,
        step=1,
        description='Workers:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Gabungkan dalam container
    return widgets.VBox([
        img_size_slider,
        enable_norm_checkbox,
        preserve_ratio_checkbox,
        enable_cache_checkbox,
        num_workers_slider
    ])
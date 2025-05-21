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
        layout=widgets.Layout(width='95%')
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
        layout=widgets.Layout(width='95%')
    )
    
    # Split selector untuk kolom kedua
    split_dropdown = widgets.Dropdown(
        options=['Train', 'Validation', 'Test', 'All'],
        value='Train',
        description='Target Split:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Buat layout 2 kolom untuk input options
    left_column = widgets.VBox([
        img_size_slider,
        enable_norm_checkbox,
        preserve_ratio_checkbox
    ], layout=widgets.Layout(width='50%'))
    
    right_column = widgets.VBox([
        num_workers_slider,
        split_dropdown,
        enable_cache_checkbox
    ], layout=widgets.Layout(width='50%'))
    
    # Gabungkan dalam container horizontal
    options_container = widgets.HBox([left_column, right_column], 
                                    layout=widgets.Layout(width='100%'))
    
    # Tambahkan atribut yang diperlukan untuk akses dari luar
    options_container.resolution = img_size_slider
    options_container.normalization = enable_norm_checkbox
    options_container.preserve_aspect_ratio = preserve_ratio_checkbox
    options_container.enable_cache = enable_cache_checkbox
    options_container.num_workers = num_workers_slider
    options_container.target_split = split_dropdown
    
    return options_container
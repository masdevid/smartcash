"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Komponen input options untuk preprocessing
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional


def create_preprocessing_input_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen input options untuk preprocessing.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        widgets.VBox: Container input options
    """
    # Default config
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    
    # Extract nilai dari config dengan fallback
    img_size = preprocessing_config.get('img_size', (640, 640))
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        resolution_str = f"{img_size[0]}x{img_size[1]}"
    else:
        resolution_str = "640x640"
    
    normalization = preprocessing_config.get('normalization', 'minmax')
    preserve_aspect_ratio = preprocessing_config.get('preserve_aspect_ratio', True)
    augmentation = preprocessing_config.get('augmentation', False)
    force_reprocess = preprocessing_config.get('force_reprocess', False)
    
    # Resolution dropdown
    resolution_dropdown = widgets.Dropdown(
        options=['320x320', '416x416', '512x512', '640x640', '768x768', '896x896', '1024x1024'],
        value=resolution_str if resolution_str in ['320x320', '416x416', '512x512', '640x640', '768x768', '896x896', '1024x1024'] else '640x640',
        description='Resolusi:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Normalization dropdown
    normalization_dropdown = widgets.Dropdown(
        options=['none', 'minmax', 'standard', 'robust', 'yolo'],
        value=normalization if normalization in ['none', 'minmax', 'standard', 'robust', 'yolo'] else 'minmax',
        description='Normalisasi:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Checkboxes
    preserve_aspect_ratio_checkbox = widgets.Checkbox(
        value=preserve_aspect_ratio,
        description='Pertahankan aspect ratio',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    augmentation_checkbox = widgets.Checkbox(
        value=augmentation,
        description='Aktifkan augmentasi',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    force_reprocess_checkbox = widgets.Checkbox(
        value=force_reprocess,
        description='Paksa reprocess semua',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Container
    options_container = widgets.VBox([
        widgets.HTML("<h5 style='margin-bottom: 10px;'>Opsi Preprocessing</h5>"),
        resolution_dropdown,
        normalization_dropdown,
        preserve_aspect_ratio_checkbox,
        augmentation_checkbox,
        force_reprocess_checkbox
    ], layout=widgets.Layout(padding='10px 5px', width='48%'))
    
    # Tambahkan referensi untuk akses mudah
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.preserve_aspect_ratio_checkbox = preserve_aspect_ratio_checkbox
    options_container.augmentation_checkbox = augmentation_checkbox
    options_container.force_reprocess_checkbox = force_reprocess_checkbox
    
    return options_container
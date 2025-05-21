"""
File: smartcash/ui/dataset/preprocessing/components/input_options.py
Deskripsi: Komponen opsi input untuk preprocessing dataset dengan perbaikan default resolusi
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
    if not config:
        config = {}
    
    preprocessing_config = config.get('preprocessing', {})
    
    # Ekstrak image size (array atau integer)
    # Default dari dataset_constants jika tidak ada
    img_size = DEFAULT_IMG_SIZE  # Default (640, 640)
    
    if 'img_size' in preprocessing_config:
        img_size_val = preprocessing_config['img_size']
        if isinstance(img_size_val, (list, tuple)) and len(img_size_val) == 2:
            img_size = img_size_val
        elif isinstance(img_size_val, int):
            img_size = (img_size_val, img_size_val)
    
    # Konversi tuple resolusi ke string format
    resolution_str = f"{img_size[0]}x{img_size[1]}"
    
    # Ekstrak opsi normalisasi
    normalization = preprocessing_config.get('normalization', 'minmax')
    preserve_aspect_ratio = preprocessing_config.get('preserve_aspect_ratio', True)
    augmentation = preprocessing_config.get('augmentation', False)
    force_reprocess = preprocessing_config.get('force_reprocess', False)
    
    # Buat dropdown untuk resolusi - format harus str
    resolution_options = ['320x320', '416x416', '512x512', '640x640', '768x768', '896x896', '1024x1024']
    
    # Pastikan resolusi yang digunakan adalah salah satu dari opsi yang ada
    if resolution_str not in resolution_options:
        resolution_str = '640x640'  # Fallback ke default jika tidak valid
    
    resolution_dropdown = widgets.Dropdown(
        options=resolution_options,
        value=resolution_str,
        description='Resolusi:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Buat dropdown untuk normalisasi
    normalization_options = ['none', 'minmax', 'standard', 'robust', 'yolo']
    
    # Pastikan normalisasi yang digunakan adalah salah satu dari opsi yang ada
    if normalization not in normalization_options:
        normalization = 'minmax'  # Fallback ke default jika tidak valid
    
    normalization_dropdown = widgets.Dropdown(
        options=normalization_options,
        value=normalization,
        description='Normalisasi:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Buat checkbox untuk preserve aspect ratio
    preserve_aspect_ratio_checkbox = widgets.Checkbox(
        value=preserve_aspect_ratio,
        description='Pertahankan aspect ratio',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Buat checkbox untuk augmentasi
    augmentation_checkbox = widgets.Checkbox(
        value=augmentation,
        description='Aktifkan augmentasi',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Buat checkbox untuk force reprocess
    force_reprocess_checkbox = widgets.Checkbox(
        value=force_reprocess,
        description='Paksa reprocess semua',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='95%')
    )
    
    # Container untuk opsi preprocessing utama
    options_container = widgets.VBox([
        widgets.HTML("<h5 style='margin-bottom: 10px;'>Opsi Preprocessing</h5>"),
        resolution_dropdown,
        normalization_dropdown,
        preserve_aspect_ratio_checkbox,
        augmentation_checkbox,
        force_reprocess_checkbox
    ], layout=widgets.Layout(padding='10px 5px', width='48%'))
    
    # Tambahkan atribut yang mempermudah akses komponen dari luar
    options_container.resolution_dropdown = resolution_dropdown
    options_container.normalization_dropdown = normalization_dropdown
    options_container.preserve_aspect_ratio_checkbox = preserve_aspect_ratio_checkbox
    options_container.augmentation_checkbox = augmentation_checkbox
    options_container.force_reprocess_checkbox = force_reprocess_checkbox
    
    return options_container
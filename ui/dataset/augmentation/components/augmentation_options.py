"""
File: smartcash/ui/dataset/augmentation/components/augmentation_options.py
Deskripsi: Komponen opsi augmentasi untuk augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List

def create_augmentation_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi augmentasi.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi augmentasi
    """
    # Dapatkan nilai default dari config jika tersedia
    augmentations = ['Combined (Recommended)']  # Default sebagai list
    augmentation_factor = 2
    aug_prefix = 'aug'
    target_split = 'train'
    balance_classes = True
    num_workers = 4
    
    if config and 'augmentation' in config:
        aug_config = config['augmentation']
        
        # Pastikan types selalu dalam format yang benar
        if 'types' in aug_config:
            # Jika nilai dari config adalah list, konversi ke tuple
            if isinstance(aug_config['types'], list):
                augmentations = tuple(aug_config['types'])
            # Jika nilai dari config adalah string tunggal, buat tuple dengan satu item
            elif isinstance(aug_config['types'], str):
                augmentations = (aug_config['types'],)
            # Jika nilai dari config sudah tuple, gunakan langsung
            elif isinstance(aug_config['types'], tuple):
                augmentations = aug_config['types']
        
        augmentation_factor = aug_config.get('factor', augmentation_factor)
        aug_prefix = aug_config.get('prefix', aug_prefix)
        target_split = aug_config.get('target_split', target_split)
        balance_classes = aug_config.get('balance_classes', balance_classes)
        num_workers = aug_config.get('num_workers', num_workers)
    
    # Buat komponen-komponen UI
    aug_type_selector = widgets.SelectMultiple(
        options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
        value=['Combined (Recommended)'],  # Gunakan list untuk nilai default
        description='Jenis:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%', height='80px')
    )
    
    # Set nilai dari config jika tersedia
    try:
        valid_options = ['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation']
        
        # Validasi nilai augmentations
        if isinstance(augmentations, (list, tuple)) and len(augmentations) > 0:
            # Filter nilai yang valid
            valid_values = [val for val in augmentations if val in valid_options]
            if valid_values:
                # Pastikan nilai adalah list
                aug_type_selector.value = valid_values
            else:
                # Jika tidak ada nilai valid, gunakan default
                aug_type_selector.value = ['Combined (Recommended)']
        else:
            # Jika augmentations bukan list atau tuple, gunakan default
            aug_type_selector.value = ['Combined (Recommended)']
    except Exception as e:
        # Jika terjadi error, gunakan nilai default
        aug_type_selector.value = ['Combined (Recommended)']
    
    factor_slider = widgets.IntSlider(
        value=augmentation_factor,
        min=1,
        max=10,
        step=1,
        description='Faktor:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    prefix_text = widgets.Text(
        value=aug_prefix,
        description='File prefix:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    split_dropdown = widgets.Dropdown(
        options=['train', 'valid', 'test', 'all'],
        value=target_split,
        description='Target split:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    balance_checkbox = widgets.Checkbox(
        value=balance_classes,
        description='Balance kelas',
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
    
    # Deskripsi singkat
    description = widgets.HTML(
        value=f"""<div style="margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;">
            <p style="margin: 0;"><strong>Augmentasi</strong> akan memperbanyak data training dengan variasi posisi, 
            pencahayaan, dan rotasi untuk meningkatkan akurasi model.</p>
        </div>"""
    )
    
    # Gabungkan dalam container
    return widgets.VBox([
        description,
        aug_type_selector,
        prefix_text,
        factor_slider,
        split_dropdown,
        balance_checkbox,
        num_workers_slider
    ])
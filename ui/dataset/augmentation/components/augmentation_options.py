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
    augmentations = ['Combined (Recommended)']
    augmentation_factor = 2
    aug_prefix = 'aug'
    target_split = 'train'
    balance_classes = True
    num_workers = 4
    
    if config and 'augmentation' in config:
        aug_config = config['augmentation']
        augmentations = aug_config.get('types', augmentations)
        augmentation_factor = aug_config.get('factor', augmentation_factor)
        aug_prefix = aug_config.get('prefix', aug_prefix)
        target_split = aug_config.get('target_split', target_split)
        balance_classes = aug_config.get('balance_classes', balance_classes)
        num_workers = aug_config.get('num_workers', num_workers)
    
    # Buat komponen-komponen UI
    aug_type_selector = widgets.SelectMultiple(
        options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
        value=['Combined (Recommended)'],  # Selalu gunakan list dengan nilai default
        description='Jenis:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%', height='80px')
    )
    
    # Set nilai dari config jika tersedia
    try:
        if isinstance(augmentations, list) and all(isinstance(item, str) for item in augmentations):
            # Pastikan semua nilai dalam list adalah string dan ada dalam options
            valid_options = ['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation']
            valid_values = [val for val in augmentations if val in valid_options]
            if valid_values:
                aug_type_selector.value = valid_values
    except Exception:
        # Jika terjadi error, gunakan nilai default
        pass
    
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
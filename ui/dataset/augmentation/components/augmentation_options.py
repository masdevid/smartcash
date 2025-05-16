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
    num_variations = 2
    output_prefix = 'aug'
    balance_classes = True
    num_workers = 4
    target_count = 1000
    move_to_preprocessed = True
    
    # Nilai default untuk augmentasi
    aug_type = 'combined'
    target_split = 'train'
    
    # Opsi yang tersedia
    available_aug_types = ['combined', 'flip', 'rotate', 'blur', 'noise', 'contrast', 'brightness', 'saturation', 'hue', 'cutout']
    available_splits = ['train', 'valid', 'test']
    
    if config and 'augmentation' in config:
        aug_config = config['augmentation']
        num_variations = aug_config.get('num_variations', num_variations)
        output_prefix = aug_config.get('output_prefix', output_prefix)
        balance_classes = aug_config.get('balance_classes', balance_classes)
        num_workers = aug_config.get('num_workers', num_workers)
        target_count = aug_config.get('target_count', target_count)
        move_to_preprocessed = aug_config.get('move_to_preprocessed', move_to_preprocessed)
        
        # Ambil nilai dari config jika tersedia
        if 'types' in aug_config and isinstance(aug_config['types'], list) and aug_config['types']:
            aug_type = aug_config['types'][0]
        
        if 'split' in aug_config:
            target_split = aug_config['split']
            
        # Pastikan nilai valid
        if aug_type not in available_aug_types:
            aug_type = 'combined'
            
        if target_split not in available_splits:
            target_split = 'train'
    
    # Buat tab untuk opsi dasar dan opsi lanjutan
    basic_tab = widgets.VBox()
    advanced_tab = widgets.VBox()
    
    # Tab 1: Opsi Dasar
    factor_slider = widgets.IntSlider(
        value=num_variations,
        min=1,
        max=10,
        step=1,
        description='Jumlah variasi:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    prefix_text = widgets.Text(
        value=output_prefix,
        description='File prefix:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    # Dropdown untuk memilih jenis augmentasi
    aug_types_dropdown = widgets.Dropdown(
        options=[
            ('Combined (Recommended)', 'combined'),
            ('Flip', 'flip'),
            ('Rotate', 'rotate'),
            ('Blur', 'blur'),
            ('Noise', 'noise'),
            ('Contrast', 'contrast'),
            ('Brightness', 'brightness'),
            ('Saturation', 'saturation'),
            ('Hue', 'hue'),
            ('Cutout', 'cutout')
        ],
        value=aug_type,
        description='Jenis augmentasi:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    # Dropdown untuk memilih target split
    split_dropdown = widgets.Dropdown(
        options=[
            ('Train', 'train'),
            ('Validation', 'valid'),
            ('Test', 'test')
        ],
        value=target_split,
        description='Target split:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    # Tab 2: Opsi Lanjutan
    balance_checkbox = widgets.Checkbox(
        value=balance_classes,
        description='Balance kelas',
        style={'description_width': 'initial'}
    )
    
    target_count_slider = widgets.IntSlider(
        value=target_count,
        min=100,
        max=5000,
        step=100,
        description='Target per kelas:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    num_workers_slider = widgets.IntSlider(
        value=num_workers,
        min=1,
        max=16,
        step=1,
        description='Workers:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    move_to_preprocessed_checkbox = widgets.Checkbox(
        value=move_to_preprocessed,
        description='Pindahkan ke preprocessed',
        style={'description_width': 'initial'}
    )
    
    # Deskripsi singkat dengan informasi tambahan tentang default values
    description = widgets.HTML(
        value=f"""<div style="margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;">
            <p style="margin: 0;"><strong>Augmentasi</strong> akan memperbanyak data training dengan kombinasi variasi posisi, 
            pencahayaan, dan rotasi untuk meningkatkan akurasi model.</p>
        </div>"""
    )
    
    # Isi tab dasar
    basic_tab.children = [
        aug_types_dropdown,
        prefix_text,
        factor_slider,
        split_dropdown
    ]
    
    # Isi tab lanjutan
    advanced_tab.children = [
        balance_checkbox,
        target_count_slider,
        num_workers_slider,
        move_to_preprocessed_checkbox
    ]
    
    # Buat tab container
    tab = widgets.Tab()
    tab.children = [basic_tab, advanced_tab]
    tab.set_title(0, 'Opsi Dasar')
    tab.set_title(1, 'Opsi Lanjutan')
    
    # Gabungkan dalam container dengan komponen yang diperbarui
    result = widgets.VBox([
        description,
        tab
    ])
    
    # Simpan referensi ke komponen UI untuk diakses oleh handler
    result.aug_types_dropdown = aug_types_dropdown
    result.split_dropdown = split_dropdown
    result.prefix_text = prefix_text
    result.factor_slider = factor_slider
    result.balance_checkbox = balance_checkbox
    result.target_count_slider = target_count_slider
    result.num_workers_slider = num_workers_slider
    result.move_to_preprocessed_checkbox = move_to_preprocessed_checkbox
    
    return result
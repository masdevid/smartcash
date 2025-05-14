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
    augmentation_factor = 2
    aug_prefix = 'aug'
    balance_classes = True
    num_workers = 4
    
    # Nilai tetap untuk augmentasi
    augmentations = ['Combined (Recommended)']  # Selalu menggunakan Combined
    target_split = 'train'  # Selalu menggunakan train split
    
    if config and 'augmentation' in config:
        aug_config = config['augmentation']
        augmentation_factor = aug_config.get('factor', augmentation_factor)
        aug_prefix = aug_config.get('prefix', aug_prefix)
        balance_classes = aug_config.get('balance_classes', balance_classes)
        num_workers = aug_config.get('num_workers', num_workers)
        
        # Simpan nilai default untuk jenis augmentasi dan target split
        aug_config['types'] = augmentations
        aug_config['split'] = target_split
    
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
    
    # Informasi tentang target split (tidak lagi sebagai dropdown)
    split_info = widgets.HTML(
        value=f"<div style='padding: 5px; margin: 5px 0;'><b>Target split:</b> train (default)</div>",
        layout=widgets.Layout(width='50%')
    )
    
    # Informasi tentang jenis augmentasi (tidak lagi sebagai selector)
    aug_type_info = widgets.HTML(
        value=f"<div style='padding: 5px; margin: 5px 0;'><b>Jenis augmentasi:</b> Combined (Recommended)</div>",
        layout=widgets.Layout(width='70%')
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
    
    # Deskripsi singkat dengan informasi tambahan tentang default values
    description = widgets.HTML(
        value=f"""<div style="margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-left: 3px solid #5bc0de;">
            <p style="margin: 0;"><strong>Augmentasi</strong> akan memperbanyak data training dengan kombinasi variasi posisi, 
            pencahayaan, dan rotasi untuk meningkatkan akurasi model.</p>
        </div>"""
    )
    
    # Gabungkan dalam container dengan komponen yang diperbarui
    return widgets.VBox([
        description,
        aug_type_info,  # Informasi jenis augmentasi (bukan selector)
        prefix_text,
        factor_slider,
        split_info,     # Informasi target split (bukan dropdown)
        balance_checkbox,
        num_workers_slider
    ])
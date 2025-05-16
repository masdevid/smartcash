"""
File: smartcash/ui/dataset/augmentation/components/augmentation_options.py
Deskripsi: Komponen UI untuk opsi augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

def create_augmentation_options(config: Dict[str, Any] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi augmentasi dataset.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi augmentasi
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.common.config.manager import get_config_manager
    
    # Dapatkan konfigurasi augmentasi
    config_manager = get_config_manager()
    aug_config = config_manager.get_module_config('augmentation')
    
    # Daftar jenis augmentasi yang tersedia
    available_types = [
        'combined',  # Kombinasi posisi dan pencahayaan (direkomendasikan)
        'position',  # Variasi posisi seperti rotasi, flipping, dan scaling
        'lighting'   # Variasi pencahayaan seperti brightness, contrast dan HSV
    ]
    
    # Daftar split yang tersedia
    available_splits = ['train', 'valid', 'test']
    
    # Opsi dasar
    aug_enabled = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('enabled', True),
        description='Aktifkan Augmentasi',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Jumlah variasi per gambar
    num_variations = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('num_variations', 2),
        min=1,
        max=10,
        step=1,
        description='Jumlah Variasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Jenis augmentasi (multi-select) dengan deskripsi
    aug_types_options = [
        ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
        ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
        ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
    ]
    
    aug_types = widgets.SelectMultiple(
        options=aug_types_options,
        value=aug_config.get('augmentation', {}).get('types', ['combined']),
        description='Jenis Augmentasi:',
        disabled=False,
        layout=widgets.Layout(width='95%', height='120px')
    )
    
    # Target split
    target_split = widgets.Dropdown(
        options=available_splits,
        value=aug_config.get('augmentation', {}).get('target_split', 'train'),
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Prefix output
    output_prefix = widgets.Text(
        value=aug_config.get('augmentation', {}).get('output_prefix', 'aug'),
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Target jumlah per kelas
    target_count = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('target_count', 1000),
        min=100,
        max=5000,
        step=100,
        description='Target per Kelas:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Balancing kelas
    balance_classes = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('balance_classes', True),
        description='Balancing Kelas',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Pindahkan ke preprocessed
    move_to_preprocessed = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('move_to_preprocessed', True),
        description='Pindahkan ke Preprocessed',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Validasi hasil
    validate_results = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('validate_results', True),
        description='Validasi Hasil',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Resume augmentasi
    resume = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('resume', False),
        description='Resume Augmentasi',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Jumlah workers
    num_workers = widgets.IntSlider(
        value=aug_config.get('augmentation', {}).get('num_workers', 4),
        min=1,
        max=16,
        step=1,
        description='Jumlah Workers:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Layout opsi dasar
    basic_options = widgets.VBox([
        widgets.HBox([aug_enabled, balance_classes], layout=widgets.Layout(justify_content='space-between')),
        widgets.HBox([move_to_preprocessed, validate_results], layout=widgets.Layout(justify_content='space-between')),
        widgets.HBox([resume], layout=widgets.Layout(justify_content='flex-start')),
        num_variations,
        target_count,
        num_workers,
        output_prefix
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Layout jenis augmentasi
    augmentation_types_box = widgets.VBox([
        aug_types,
        target_split
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Tab untuk opsi dasar dan jenis augmentasi
    tabs = widgets.Tab(children=[basic_options, augmentation_types_box])
    tabs.set_title(0, f"{ICONS['settings']} Opsi Dasar")
    tabs.set_title(1, f"{ICONS['augmentation']} Jenis Augmentasi")
    
    # Container utama
    container = widgets.VBox([
        tabs
    ], layout=widgets.Layout(margin='10px 0'))
    
    return container

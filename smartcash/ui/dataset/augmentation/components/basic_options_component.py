"""
File: smartcash/ui/dataset/augmentation/components/basic_options_component.py
Deskripsi: Komponen UI untuk opsi dasar augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_basic_options_component(config: Dict[str, Any] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi dasar augmentasi dataset.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi dasar augmentasi
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.common.config.manager import get_config_manager
    from smartcash.common.logger import get_logger
    
    logger = get_logger('augmentation')
    
    try:
        # Dapatkan konfigurasi augmentasi
        config_manager = get_config_manager()
        if config_manager is None:
            aug_config = {}
        else:
            aug_config = config_manager.get_module_config('augmentation')
        
        # Gunakan config yang diberikan jika ada
        if config is not None and isinstance(config, dict):
            if 'augmentation' in config:
                aug_config = config.get('augmentation', {})
        
        # Pastikan aug_config memiliki struktur yang benar
        if not aug_config or not isinstance(aug_config, dict):
            aug_config = {}
        if 'augmentation' not in aug_config:
            aug_config['augmentation'] = {}
    except Exception as e:
        aug_config = {}
        aug_config['augmentation'] = {}
    
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
    
    # Target count
    target_count = widgets.IntText(
        value=aug_config.get('augmentation', {}).get('target_count', 0),
        description='Target Count:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Prefix output
    output_prefix = widgets.Text(
        value=aug_config.get('augmentation', {}).get('output_prefix', 'aug_'),
        placeholder='Prefix untuk file hasil augmentasi',
        description='Output Prefix:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Balance classes
    balance_classes = widgets.Checkbox(
        value=aug_config.get('augmentation', {}).get('balance_classes', False),
        description='Balance Classes',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Container untuk opsi dasar
    basic_options_container = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['settings']} Opsi Dasar</h5>"),
        num_variations,
        target_count,
        output_prefix,
        widgets.HBox([aug_enabled, balance_classes], layout=widgets.Layout(justify_content='space-between'))
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    return basic_options_container

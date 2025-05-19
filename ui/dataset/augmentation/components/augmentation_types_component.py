"""
File: smartcash/ui/dataset/augmentation/components/augmentation_types_component.py
Deskripsi: Komponen UI untuk jenis augmentasi dan target split dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_augmentation_types_component(config: Dict[str, Any] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk jenis augmentasi dan target split dataset.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi jenis augmentasi dan target split
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
    
    # Daftar jenis augmentasi yang tersedia
    aug_types_options = [
        ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
        ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
        ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
    ]
    
    # Dapatkan nilai types dari konfigurasi atau gunakan default
    aug_types_value = aug_config.get('augmentation', {}).get('types', ['combined'])
    
    # Validasi nilai aug_types terhadap opsi yang tersedia
    available_values = [opt[1] for opt in aug_types_options]
    valid_aug_types = [t for t in aug_types_value if t in available_values]
    
    # Jika tidak ada nilai valid, gunakan default
    if not valid_aug_types:
        valid_aug_types = ['combined']
    
    # Jenis augmentasi (multi-select) dengan deskripsi
    aug_types = widgets.SelectMultiple(
        options=aug_types_options,
        value=valid_aug_types,
        description='Jenis:',
        disabled=False,
        layout=widgets.Layout(width='95%', height='100px')
    )
    
    # Daftar split yang tersedia
    available_splits = ['train', 'valid', 'test']
    
    # Target split
    target_split = widgets.Dropdown(
        options=available_splits,
        value=aug_config.get('augmentation', {}).get('target_split', 'train'),
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Checkbox telah dipindahkan ke basic_options_component atau dihilangkan
    
    # Informasi tentang split
    split_info = widgets.HTML(
        f"""
        <div style="padding: 5px; color: {COLORS['dark']};">
            <p><b>{ICONS['info']} Informasi Split:</b></p>
            <ul>
                <li><b>train</b>: Augmentasi pada data training (rekomendasi)</li>
                <li><b>valid</b>: Augmentasi pada data validasi (jarang diperlukan)</li>
                <li><b>test</b>: Augmentasi pada data testing (tidak direkomendasikan)</li>
            </ul>
        </div>
        """
    )
    
    # Container untuk jenis augmentasi
    augmentation_types_container = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['augmentation']} Jenis Augmentasi & Target Split</h5>"),
        widgets.HBox([
            # Kolom kiri: Jenis augmentasi
            widgets.VBox([
                widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>Pilih Jenis Augmentasi:</h6>"),
                aug_types
            ], layout=widgets.Layout(width='60%')),
            
            # Kolom kanan: Target split dan info
            widgets.VBox([
                widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['split']} Target Split:</h6>"),
                target_split,
                split_info
            ], layout=widgets.Layout(width='40%'))
        ], layout=widgets.Layout(width='100%'))
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    return augmentation_types_container

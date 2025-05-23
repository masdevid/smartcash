"""
File: smartcash/ui/dataset/preprocessing/components/config_manager.py
Deskripsi: Manajemen konfigurasi untuk UI preprocessing dengan parameter sesuai backend
"""

from typing import Dict, Any
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE


def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update UI components dari konfigurasi dengan parameter sesuai backend.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi dari config manager
    """
    try:
        # Extract preprocessing config
        preprocessing_config = config.get('preprocessing', {})
        
        # Update resolution
        img_size = preprocessing_config.get('img_size', DEFAULT_IMG_SIZE)
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            resolution_str = f"{img_size[0]}x{img_size[1]}"
            if 'resolution_dropdown' in ui_components:
                dropdown = ui_components['resolution_dropdown']
                if resolution_str in dropdown.options:
                    dropdown.value = resolution_str
        
        # Update normalization
        normalization = preprocessing_config.get('normalization', 'minmax')
        if 'normalization_dropdown' in ui_components:
            dropdown = ui_components['normalization_dropdown']
            if normalization in dropdown.options:
                dropdown.value = normalization
        
        # Update worker slider
        if 'worker_slider' in ui_components:
            num_workers = preprocessing_config.get('num_workers', 4)
            ui_components['worker_slider'].value = min(max(num_workers, 1), 10)
        
        # Update split dropdown
        if 'split_dropdown' in ui_components:
            split = preprocessing_config.get('split', 'all')
            if split in ui_components['split_dropdown'].options:
                ui_components['split_dropdown'].value = split
        
    except Exception as e:
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"⚠️ Error update UI dari config: {str(e)}")


def get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract konfigurasi dari UI components untuk backend preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict[str, Any]: Konfigurasi preprocessing sesuai backend
    """
    # Base config dengan default values sesuai backend
    config = {
        'preprocessing': {
            'img_size': DEFAULT_IMG_SIZE,
            'normalization': 'minmax',
            'normalize': True,
            'preserve_aspect_ratio': True,  # Fixed True
            'augmentation': False,  # Fixed False  
            'force_reprocess': True,  # Fixed True
            'num_workers': 4,
            'split': 'all',
            'output_dir': 'data/preprocessed'
        },
        'data': {
            'dir': 'data'
        }
    }
    
    try:
        preprocessing_config = config['preprocessing']
        
        # Extract resolution
        if 'resolution_dropdown' in ui_components:
            resolution_str = ui_components['resolution_dropdown'].value
            if 'x' in resolution_str:
                width, height = map(int, resolution_str.split('x'))
                preprocessing_config['img_size'] = (width, height)
        
        # Extract normalization
        if 'normalization_dropdown' in ui_components:
            normalization = ui_components['normalization_dropdown'].value
            preprocessing_config['normalization'] = normalization
            preprocessing_config['normalize'] = normalization != 'none'
        
        # Extract worker count
        if 'worker_slider' in ui_components:
            preprocessing_config['num_workers'] = ui_components['worker_slider'].value
        
        # Extract split
        if 'split_dropdown' in ui_components:
            preprocessing_config['split'] = ui_components['split_dropdown'].value
        
        # Update paths dari UI components
        preprocessing_config['output_dir'] = ui_components.get('preprocessed_dir', 'data/preprocessed')
        config['data']['dir'] = ui_components.get('data_dir', 'data')
        
    except Exception as e:
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(f"⚠️ Error extract config dari UI: {str(e)}")
    
    return config
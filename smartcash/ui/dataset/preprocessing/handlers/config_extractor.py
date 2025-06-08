"""
File: smartcash/ui/dataset/preprocessing/handlers/config_extractor.py
Deskripsi: Extract config dengan DRY approach - base dari defaults lalu update form values
"""

from typing import Dict, Any

def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dengan DRY approach - base dari defaults + form values"""
    from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
    
    # Base structure dari defaults
    config = get_default_preprocessing_config()
    
    # Helper untuk get form values
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Update hanya nilai dari form
    resolution = get_value('resolution_dropdown', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    normalization_method = get_value('normalization_dropdown', 'minmax')
    
    # Update form-controlled values
    config['preprocessing']['target_split'] = get_value('split_dropdown', 'all')
    config['preprocessing']['normalization']['enabled'] = normalization_method != 'none'
    config['preprocessing']['normalization']['method'] = normalization_method if normalization_method != 'none' else 'minmax'
    config['preprocessing']['normalization']['target_size'] = [width, height]
    config['performance']['num_workers'] = get_value('worker_slider', 8)
    
    return config
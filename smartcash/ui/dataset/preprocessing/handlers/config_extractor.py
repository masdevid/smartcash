"""
File: smartcash/ui/dataset/preprocessing/handlers/config_extractor.py
Deskripsi: Enhanced DRY extraction dengan multi-split, validasi, dan aspect ratio support
"""

from typing import Dict, Any, List

def extract_preprocessing_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced DRY extraction - base dari defaults + form values"""
    from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
    
    # Base structure dari defaults (DRY)
    config = get_default_preprocessing_config()
    
    # Helper untuk get form values dengan safe fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    get_selected = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Extract resolution dari dropdown
    resolution = get_value('resolution_dropdown', '640x640')
    width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
    
    # Extract normalization method
    normalization_method = get_value('normalization_dropdown', 'minmax')
    
    # Extract multi-select target splits
    target_splits_widget = ui_components.get('target_splits_select')
    if target_splits_widget and hasattr(target_splits_widget, 'value'):
        target_splits = list(target_splits_widget.value) if target_splits_widget.value else ['train', 'valid']
    else:
        target_splits = ['train', 'valid']
    
    # Extract batch size
    batch_size = get_value('batch_size_input', 32)
    batch_size = max(1, min(batch_size, 128)) if isinstance(batch_size, int) else 32
    
    # Extract preserve aspect ratio checkbox
    preserve_aspect_ratio = get_value('preserve_aspect_checkbox', True)
    
    # Extract validation settings
    validation_enabled = get_value('validation_checkbox', True)
    move_invalid = get_value('move_invalid_checkbox', True)
    invalid_dir = get_value('invalid_dir_input', 'data/invalid').strip() or 'data/invalid'
    
    # Update form-controlled values dalam config
    config['preprocessing']['target_splits'] = target_splits
    config['preprocessing']['normalization']['enabled'] = normalization_method != 'none'
    config['preprocessing']['normalization']['method'] = normalization_method if normalization_method != 'none' else 'minmax'
    config['preprocessing']['normalization']['target_size'] = [width, height]
    config['preprocessing']['normalization']['preserve_aspect_ratio'] = preserve_aspect_ratio
    
    # Update validation settings
    config['preprocessing']['validation']['enabled'] = validation_enabled
    config['preprocessing']['validation']['move_invalid'] = move_invalid
    config['preprocessing']['validation']['invalid_dir'] = invalid_dir
    
    # Update performance settings
    config['performance']['batch_size'] = batch_size
    
    return config
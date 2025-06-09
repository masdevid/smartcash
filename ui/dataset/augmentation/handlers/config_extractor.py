"""
File: smartcash/ui/dataset/augmentation/handlers/config_extractor.py
Deskripsi: Config extractor dengan backend integration dan DRY approach
"""

from typing import Dict, Any

def extract_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract config dengan backend integration dan DRY approach
    Base dari defaults + form values untuk backend compatibility
    """
    from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
    
    # Base structure dari defaults (DRY)
    config = get_default_augmentation_config()
    
    # Helper untuk safe form value extraction
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Basic options extraction
    aug_config = config['augmentation']
    aug_config['num_variations'] = get_value('num_variations', 2)
    aug_config['target_count'] = get_value('target_count', 500)
    aug_config['output_prefix'] = get_value('output_prefix', 'aug')
    aug_config['balance_classes'] = get_value('balance_classes', True)
    aug_config['target_split'] = get_value('target_split', 'train')
    
    # Augmentation types dengan validation
    aug_types = get_value('augmentation_types', ['combined'])
    if isinstance(aug_types, (list, tuple)) and aug_types:
        aug_config['types'] = list(aug_types)
    
    # Position parameters (mapping dari UI ke backend format)
    aug_config['position']['horizontal_flip'] = get_value('fliplr', 0.5)
    aug_config['position']['rotation_limit'] = get_value('degrees', 12)
    aug_config['position']['translate_limit'] = get_value('translate', 0.08)
    aug_config['position']['scale_limit'] = get_value('scale', 0.04)
    
    # Lighting parameters  
    aug_config['lighting']['brightness_limit'] = get_value('brightness', 0.2)
    aug_config['lighting']['contrast_limit'] = get_value('contrast', 0.15)
    
    # Combined parameters (sync dengan position dan lighting)
    combined = aug_config['combined']
    combined['horizontal_flip'] = aug_config['position']['horizontal_flip']
    combined['rotation_limit'] = aug_config['position']['rotation_limit']
    combined['translate_limit'] = aug_config['position']['translate_limit']
    combined['scale_limit'] = aug_config['position']['scale_limit']
    combined['brightness_limit'] = aug_config['lighting']['brightness_limit']
    combined['contrast_limit'] = aug_config['lighting']['contrast_limit']
    
    # Preprocessing/Normalization
    preprocessing = config['preprocessing']['normalization']
    preprocessing['method'] = get_value('norm_method', 'minmax')
    preprocessing['denormalize'] = get_value('denormalize', False)
    
    # Backend integration metadata
    config['backend'].update({
        'extracted_at': _get_timestamp(),
        'ui_version': '2.0',
        'form_validated': _validate_extracted_config(config)
    })
    
    return config

def _validate_extracted_config(config: Dict[str, Any]) -> bool:
    """Validate extracted config untuk backend compatibility"""
    aug_config = config.get('augmentation', {})
    
    # Required fields check
    required = ['num_variations', 'target_count', 'types', 'target_split']
    if not all(field in aug_config for field in required):
        return False
    
    # Range validation
    if not (1 <= aug_config.get('num_variations', 0) <= 10):
        return False
    if not (100 <= aug_config.get('target_count', 0) <= 2000):
        return False
    
    return True

def _get_timestamp() -> str:
    """Get current timestamp"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
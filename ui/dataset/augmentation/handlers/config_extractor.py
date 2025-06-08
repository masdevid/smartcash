"""
File: smartcash/ui/dataset/augmentation/handlers/config_extractor.py
Deskripsi: DRY config extraction dengan defaults sebagai base
"""

from typing import Dict, Any

def extract_augmentation_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    CRITICAL: DRY approach - base dari defaults + form values
    
    Args:
        ui_components: UI components dictionary dengan form widgets
        
    Returns:
        Config dictionary yang siap disimpan
    """
    from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
    
    # Base structure dari defaults (DRY)
    config = get_default_augmentation_config()
    
    # Helper untuk get form values dengan safe access
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Update HANYA nilai dari form - Basic options
    config['augmentation']['num_variations'] = get_value('num_variations', 3)
    config['augmentation']['target_count'] = get_value('target_count', 500)
    config['augmentation']['output_prefix'] = get_value('output_prefix', 'aug')
    config['augmentation']['balance_classes'] = get_value('balance_classes', True)
    config['augmentation']['target_split'] = get_value('target_split', 'train')
    
    # Update nilai dari form - Augmentation types
    aug_types = get_value('augmentation_types', ['combined'])
    if isinstance(aug_types, (list, tuple)) and aug_types:
        config['augmentation']['types'] = list(aug_types)
    
    # Update nilai dari form - Position parameters
    config['augmentation']['position']['fliplr'] = get_value('fliplr', 0.5)
    config['augmentation']['position']['degrees'] = get_value('degrees', 10)
    config['augmentation']['position']['translate'] = get_value('translate', 0.1)
    config['augmentation']['position']['scale'] = get_value('scale', 0.1)
    
    # Update nilai dari form - Lighting parameters
    config['augmentation']['lighting']['hsv_h'] = get_value('hsv_h', 0.015)
    config['augmentation']['lighting']['hsv_s'] = get_value('hsv_s', 0.7)
    config['augmentation']['lighting']['brightness'] = get_value('brightness', 0.2)
    config['augmentation']['lighting']['contrast'] = get_value('contrast', 0.2)
    
    return config
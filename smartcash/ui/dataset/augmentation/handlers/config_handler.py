"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Config handler dengan unified logging dan parameter alignment yang diperbaiki
"""

from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.handlers.config_handlers import ConfigHandler

class AugmentationConfigHandler(ConfigHandler):
    """Config handler untuk augmentation dengan fixed implementation"""
    def __init__(self, module_name: str = 'augmentation', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'augmentation_config.yaml'

    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        config = {}
        
        # Ekstrak basic options
        if 'num_variations' in ui_components:
            config['num_variations'] = ui_components['num_variations'].value
        if 'target_count' in ui_components:
            config['target_count'] = ui_components['target_count'].value
        if 'output_prefix' in ui_components:
            config['output_prefix'] = ui_components['output_prefix'].value
        
        # Ekstrak augmentation types
        if 'augmentation_types' in ui_components:
            config['types'] = ui_components['augmentation_types'].value
        if 'target_split' in ui_components:
            config['target_split'] = ui_components['target_split'].value
        if 'balance_classes' in ui_components:
            config['balance_classes'] = ui_components['balance_classes'].value
            
        return {'augmentation': config}

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate config untuk research compatibility"""
        try:
            aug_config = config.get('augmentation', {})
            
            if aug_config.get('num_variations', 0) <= 0:
                return False
            if aug_config.get('target_count', 0) <= 0:
                return False
            if not aug_config.get('types'):
                return False
            
            ranges = {
                'fliplr': (0.0, 1.0), 'degrees': (0, 45), 'translate': (0.0, 0.5), 'scale': (0.0, 0.5),
                'hsv_h': (0.0, 0.1), 'hsv_s': (0.0, 1.0), 'brightness': (0.0, 1.0), 'contrast': (0.0, 1.0)
            }
            
            for param, (min_val, max_val) in ranges.items():
                value = aug_config.get(param)
                if value is not None and not (min_val <= value <= max_val):
                    return False
            
            return True
            
        except Exception:
            return False

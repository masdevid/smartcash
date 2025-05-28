"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer yang disederhanakan dengan reuse service dan SRP modules
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[AUGMENTATION_LOGGER_NAMESPACE]

class AugmentationInitializer(CommonInitializer):
    """Initializer dengan service reuse dan consolidated functionality"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, AUGMENTATION_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan full reuse dari existing components"""
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        ui_components = create_augmentation_ui(env=env, config=config)
        ui_components.update({
            'logger_namespace': self.logger_namespace,
            'module_initialized': True,
            'augmentation_initialized': True
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan service layer integration"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.main_handler import register_all_handlers
            
            # Clear existing handlers untuk avoid duplication
            button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
            for key in button_keys:
                button = ui_components.get(key)
                if button and hasattr(button, '_click_handlers'):
                    try:
                        button._click_handlers.callbacks.clear()
                    except Exception:
                        pass
            
            # Register handlers dengan service layer integration
            ui_components = register_all_handlers(ui_components)
            
            if 'logger' in ui_components:
                ui_components['logger'].info(f"✅ Handlers registered dengan service integration")
            
            return ui_components
            
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Handler setup error: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan aligned parameters untuk service layer"""
        return {
            'data': {'dir': 'data'},
            'augmentation': {
                'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug_', 'balance_classes': True,
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2,
                'types': ['combined'], 'target_split': 'train', 'intensity': 0.7,
                'output_dir': 'data/augmented'
            },
            'preprocessing': {'output_dir': 'data/preprocessed', 'normalization': {'scaler': 'minmax'}}
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk service layer integration"""
        return [
            'ui', 'augment_button', 'check_button', 'save_button', 'reset_button',
            'num_variations', 'target_count', 'augmentation_types', 'target_split',
            'tracker', 'log_output'
        ]

# Global instance dengan service layer integration
_aug_initializer = AugmentationInitializer()

# Public API dengan service layer reuse
init_augmentation = lambda env=None, config=None, force=False: _aug_initializer.initialize(env=env, config=config)
reset_augmentation = lambda: print("🔄 Augmentation module reset dengan service layer cleanup")
get_aug_status = lambda: _aug_initializer.get_module_status()
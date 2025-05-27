"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Fixed initializer tanpa cache complexity dengan parameter alignment
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[AUGMENTATION_LOGGER_NAMESPACE]

class AugmentationInitializer(CommonInitializer):
    """Fixed initializer tanpa cache complexity"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, AUGMENTATION_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan aligned parameters"""
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        ui_components = create_augmentation_ui(env=env, config=config)
        ui_components.update({
            'logger_namespace': self.logger_namespace,
            'module_initialized': True,
            'augmentation_initialized': True
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers tanpa cache management"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.main_handler import register_all_handlers
            
            # Clear existing handlers
            button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
            for key in button_keys:
                button = ui_components.get(key)
                if button and hasattr(button, '_click_handlers'):
                    try:
                        button._click_handlers.callbacks.clear()
                    except Exception:
                        pass
            
            ui_components = register_all_handlers(ui_components)
            
            if 'logger' in ui_components:
                ui_components['logger'].info(f"✅ Handlers registered: {ui_components.get('handlers_registered', 0)} total")
            
            return ui_components
            
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Handler setup error: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan aligned parameters"""
        return {
            'data': {'dir': 'data'},
            'augmentation': {
                'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug_', 'balance_classes': False,
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2,
                'types': ['combined'], 'target_split': 'train', 'intensity': 0.7,
                'output_dir': 'data/augmented'
            },
            'preprocessing': {'output_dir': 'data/preprocessed'}
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components dengan aligned names"""
        return [
            'ui', 'augment_button', 'check_button', 'save_button', 'reset_button',
            'num_variations', 'target_count', 'augmentation_types', 'target_split',
            'tracker', 'log_output'
        ]
    
    def _setup_log_suppression(self) -> None:
        """Enhanced log suppression untuk augmentation"""
        super()._setup_log_suppression()
        
        augmentor_loggers = [
            'smartcash.dataset.augmentor', 'smartcash.dataset.augmentor.core',
            'smartcash.dataset.augmentor.utils', 'smartcash.dataset.augmentor.strategies',
            'smartcash.dataset.augmentor.communicator', 'albumentations', 'cv2'
        ]
        
        import logging
        for logger_name in augmentor_loggers:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
            except Exception:
                pass


# Global instance - simplified
_aug_initializer = AugmentationInitializer()

# Public API - no cache
init_augmentation = lambda env=None, config=None, force=False: _aug_initializer.initialize(env=env, config=config)
reset_augmentation = lambda: print("🔄 Augmentation module reset")  # Simple reset
get_aug_status = lambda: _aug_initializer.get_module_status()
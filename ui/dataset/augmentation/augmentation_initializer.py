"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Refactored augmentation initializer dengan komunikasi yang tepat ke augmentor service
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer

class AugmentationInitializer(CommonInitializer):
    """Refactored initializer dengan komunikasi langsung ke augmentor service"""
    
    def __init__(self):
        super().__init__(
            module_name='dataset_augmentation',
            logger_namespace='smartcash.ui.dataset.augmentation'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan setup yang minimal"""
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        ui_components = create_augmentation_ui(env=env, config=config)
        ui_components.update({
            'logger_namespace': self.logger_namespace,
            'module_initialized': True
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan reset capability"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.main_handler import register_all_handlers
            
            ui_components = register_all_handlers(ui_components)
            
            if 'logger' in ui_components:
                ui_components['logger'].info(f"✅ Handlers registered: {ui_components.get('handlers_registered', 0)} total")
            
            return ui_components
            
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Handler setup error: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan train as default"""
        return {
            'data_dir': 'data',
            'augmented_dir': 'data/augmented',
            'preprocessed_dir': 'data/preprocessed',
            'num_variations': 2,
            'target_count': 500,
            'output_prefix': 'aug',
            'balance_classes': False,
            'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2,
            'augmentation_types': ['combined'],
            'target_split': 'train',  # Default train
            'intensity': 0.7
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk validation"""
        return [
            'ui', 'augment_button', 'check_button', 'save_button', 'reset_button',
            'num_variations', 'target_count', 'augmentation_types', 'target_split',
            'tracker', 'log_output'
        ]

# Global instance
_aug_initializer = None

get_aug_initializer = lambda: globals().setdefault('_aug_initializer', AugmentationInitializer())
init_augmentation = lambda env=None, config=None, force=False: get_aug_initializer().initialize(env=env, config=config, force_refresh=force)
reset_augmentation = lambda: get_aug_initializer().reset_module()
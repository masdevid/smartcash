"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer yang disederhanakan dengan reuse service dan SRP modules
"""

from typing import Dict, Any, List, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer, create_common_initializer
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.handlers.config_handlers import ConfigHandler

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[AUGMENTATION_LOGGER_NAMESPACE]

class AugmentationConfigHandler(ConfigHandler):
    """Config handler untuk augmentation dengan fixed implementation"""
    
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
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        augmentation_config = config.get('augmentation', {})
        
        # Update basic options
        if 'num_variations' in ui_components and 'num_variations' in augmentation_config:
            ui_components['num_variations'].value = augmentation_config['num_variations']
        if 'target_count' in ui_components and 'target_count' in augmentation_config:
            ui_components['target_count'].value = augmentation_config['target_count']
        if 'output_prefix' in ui_components and 'output_prefix' in augmentation_config:
            ui_components['output_prefix'].value = augmentation_config['output_prefix']
        
        # Update augmentation types
        if 'augmentation_types' in ui_components and 'types' in augmentation_config:
            ui_components['augmentation_types'].value = augmentation_config['types']
        if 'target_split' in ui_components and 'target_split' in augmentation_config:
            ui_components['target_split'].value = augmentation_config['target_split']
        if 'balance_classes' in ui_components and 'balance_classes' in augmentation_config:
            ui_components['balance_classes'].value = augmentation_config['balance_classes']
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk augmentation"""
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

class AugmentationInitializer(CommonInitializer):
    """Initializer dengan service reuse dan consolidated functionality"""
    
    def __init__(self, module_name: str = MODULE_LOGGER_NAME, config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = 'dataset'):
        super().__init__(module_name, config_handler_class or AugmentationConfigHandler, parent_module)
    
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
        # Menggunakan config handler untuk mendapatkan default config
        config_handler = self._create_config_handler()
        return config_handler.get_default_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk service layer integration"""
        return [
            'ui', 'augment_button', 'check_button', 'save_button', 'reset_button',
            'num_variations', 'target_count', 'augmentation_types', 'target_split',
            'log_output', 'log_accordion',
            'progress_tracker', 'progress_container', 'show_for_operation', 
            'update_progress', 'complete_operation', 'error_operation', 'reset_all'
        ]

# Global instance dengan service layer integration
_aug_initializer = AugmentationInitializer()

# Public API dengan service layer reuse dan parent module support
def init_augmentation(env=None, config=None, parent_callbacks=None, **kwargs):
    """Initialize augmentation UI dengan parent module support"""
    return _aug_initializer.initialize(env=env, config=config, parent_callbacks=parent_callbacks, **kwargs)

def reset_augmentation():
    """Reset augmentation module dengan service layer cleanup"""
    print("🔄 Augmentation module reset dengan service layer cleanup")
    return _aug_initializer.reset()

def get_aug_status():
    """Get augmentation module status"""
    return _aug_initializer.get_module_status()
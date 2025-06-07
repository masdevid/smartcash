"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Initializer yang disederhanakan dengan reuse service dan SRP modules
"""

from typing import Dict, Any, List, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler

class AugmentationInitializer(CommonInitializer):
    """Initializer dengan service reuse dan consolidated functionality"""
    
    def __init__(self):
        super().__init__(
            module_name='augmentation',
            config_handler_class=AugmentationConfigHandler,
            parent_module='dataset'
        )
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan full reuse dari existing components"""
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        ui_components = create_augmentation_ui(env=env, config=config)
        ui_components.update({
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
def init_augmentation(env=None, config=None, **kwargs):
    """Initialize augmentation UI dengan parent module support"""
    return _aug_initializer.initialize(env=env, config=config, **kwargs)

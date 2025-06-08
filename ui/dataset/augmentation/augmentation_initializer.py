"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Augmentation initializer dengan CommonInitializer pattern yang diperbaiki
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler
from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
from smartcash.ui.dataset.augmentation.handlers.main_handlers import setup_augmentation_handlers

class AugmentationInitializer(CommonInitializer):
    """Augmentation initializer dengan CommonInitializer inheritance dan config management"""
    
    def __init__(self):
        super().__init__(
            module_name='augmentation',
            config_handler_class=AugmentationConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create augmentation UI components dengan config integration"""
        ui_components = create_augmentation_main_ui(config)
        ui_components.update({
            'augmentation_initialized': True,
            'module_name': 'augmentation',
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'env': env
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan auto config load dan UI update"""
        # Setup handlers terlebih dahulu
        result = setup_augmentation_handlers(ui_components, config, env)
        
        # CRITICAL: Load config dari file dan update UI
        self._load_and_update_ui(ui_components)
        
        return result
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """CRITICAL: Load config dari file dan update UI saat initialization"""
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                # Set UI components untuk logging
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                
                # Load config dari file dengan inheritance
                loaded_config = config_handler.load_config()
                
                # Update UI dengan loaded config
                config_handler.update_ui(ui_components, loaded_config)
                
                # Update config reference
                ui_components['config'] = loaded_config
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.warning(f"⚠️ Error loading config: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py"""
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        return get_default_augmentation_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk augmentation module"""
        return [
            'ui', 'augment_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'num_variations', 'target_count', 'augmentation_types'
        ]

# Global instance
_augmentation_initializer = AugmentationInitializer()

def initialize_augmentation_ui(env=None, config=None, **kwargs):
    """Factory function untuk augmentation UI dengan auto config load"""
    return _augmentation_initializer.initialize(env=env, config=config, **kwargs)
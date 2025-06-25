"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Augmentation initializer dengan backend integration dan proper progress tracking
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler
from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
from smartcash.ui.dataset.augmentation.handlers.augmentation_handlers import setup_augmentation_handlers

class AugmentationInitializer(CommonInitializer):
    """Augmentation initializer dengan backend integration dan progress tracking"""
    
    def __init__(self):
        super().__init__(
            module_name='dataset.augmentation',
            config_handler_class=AugmentationConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create augmentation UI dengan backend communicator integration"""
        ui_components = create_augmentation_main_ui(config)
        
        # Add backend integration
        ui_components.update({
            'augmentation_initialized': True,
            'module_name': 'augmentation',
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'env': env,
            'backend_ready': True
        })
        
        # Setup backend communicator
        self._setup_backend_integration(ui_components)
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan backend integration"""
        result = setup_augmentation_handlers(ui_components, config, env)
        self._load_and_update_ui(ui_components)
        return result
    
    def _setup_backend_integration(self, ui_components: Dict[str, Any]):
        """Setup backend integration tanpa external communicator"""
        ui_components.update({
            'backend_ready': True,
            'service_integration': True
        })
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """Load config dan update UI saat initialization"""
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                
                loaded_config = config_handler.load_config()
                config_handler.update_ui(ui_components, loaded_config)
                ui_components['config'] = loaded_config
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.warning(f"⚠️ Error loading config: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dari augmentation_config.yaml"""
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        return get_default_augmentation_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'augment_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'progress_tracker'
        ]

# Global instance
_augmentation_initializer = AugmentationInitializer()

def initialize_augmentation_ui(env=None, config=None, **kwargs):
    """Factory function untuk augmentation UI"""
    return _augmentation_initializer.initialize(env=env, config=config, **kwargs)
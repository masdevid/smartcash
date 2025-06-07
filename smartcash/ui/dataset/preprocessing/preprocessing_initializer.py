"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Preprocessing initializer yang mewarisi CommonInitializer dengan structure yang konsisten
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingInitializer(CommonInitializer):
    """Preprocessing initializer dengan complete UI dan backend integration"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create preprocessing UI components"""
        ui_components = create_preprocessing_main_ui(config)
        ui_components.update({
            'preprocessing_initialized': True,
            'module_name': 'preprocessing',
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'preprocessed_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan backend integration"""
        return setup_preprocessing_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk preprocessing"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'progress_container', 'show_for_operation', 
            'update_progress', 'complete_operation', 'error_operation', 'reset_all'
        ]

# Global instance
_preprocessing_initializer = PreprocessingInitializer()

def initialize_preprocessing_ui(env=None, config=None, **kwargs):
    """Factory function untuk preprocessing UI dengan parent module support"""
    return _preprocessing_initializer.initialize(env=env, config=config, **kwargs)
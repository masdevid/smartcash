"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Updated preprocessing initializer dengan critical components yang tepat
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingInitializer(CommonInitializer):
    """Optimized preprocessing initializer dengan updated critical components"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper config"""
        ui_components = create_preprocessing_main_ui(config)
        ui_components.update({
            'preprocessing_initialized': True,
            'module_name': 'preprocessing',
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'preprocessed_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan auto config load"""
        result = setup_preprocessing_handlers(ui_components, config, env)
        self._load_and_update_ui(ui_components)
        return result
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """Load config dan update UI"""
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
        """Get default config"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def _get_critical_components(self) -> List[str]:
        """Updated critical components sesuai dengan UI struktur baru"""
        return [
            'ui', 'header', 'status_panel',
            'preprocess_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button',
            'confirmation_area',
            'progress_tracker', 'progress_container',
            'log_output', 'log_accordion',
            'resolution_dropdown', 'normalization_dropdown', 
            'target_splits_select', 'batch_size_input',
            'validation_checkbox', 'preserve_aspect_checkbox'
        ]

# Global instance
_preprocessing_initializer = PreprocessingInitializer()

def initialize_preprocessing_ui(env=None, config=None, **kwargs):
    """Factory function untuk preprocessing UI"""
    return _preprocessing_initializer.initialize(env=env, config=config, **kwargs)
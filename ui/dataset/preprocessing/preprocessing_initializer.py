"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Simplified preprocessing initializer focused on critical functions
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingInitializer(CommonInitializer):
    """🎯 Simplified preprocessing initializer fokus pada fungsi critical"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """🏗️ Create UI components dengan validation minimal"""
        try:
            ui_components = create_preprocessing_main_ui(config)
            
            # Validate critical components only
            missing = [name for name in self._get_critical_components() if name not in ui_components]
            if missing:
                raise ValueError(f"Missing components: {', '.join(missing)}")
            
            ui_components.update({
                'preprocessing_initialized': True,
                'module_name': 'preprocessing',
                'data_dir': config.get('data', {}).get('dir', 'data')
            })
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ UI creation failed: {str(e)}")
            raise
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """🔧 Setup handlers dengan error handling"""
        try:
            result = setup_preprocessing_handlers(ui_components, config, env)
            self._load_and_update_ui(ui_components)
            return result
        except Exception as e:
            self.logger.error(f"❌ Handler setup failed: {str(e)}")
            return ui_components
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """📂 Load config dan update UI"""
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            loaded_config = config_handler.load_config()
            if loaded_config:
                config_handler.update_ui(ui_components, loaded_config)
                ui_components['config'] = loaded_config
                
        except Exception as e:
            self.logger.warning(f"⚠️ Config load failed: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """📋 Get default config"""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            self.logger.error(f"❌ Default config failed: {str(e)}")
            return {
                'preprocessing': {'enabled': True, 'target_splits': ['train', 'valid']},
                'performance': {'batch_size': 32},
                'data': {'dir': 'data'}
            }
    
    def _get_critical_components(self) -> List[str]:
        """📝 Critical components list"""
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel'
        ]

# Global instance
_preprocessing_initializer = PreprocessingInitializer()

def initialize_preprocessing_ui(env=None, config=None, **kwargs):
    """🏭 Factory function untuk preprocessing UI"""
    return _preprocessing_initializer.initialize(env=env, config=config, **kwargs)
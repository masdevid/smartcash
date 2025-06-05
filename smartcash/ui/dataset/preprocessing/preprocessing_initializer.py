"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer preprocessing yang terintegrasi dengan dataset/preprocessors tanpa duplikasi
"""

from typing import Dict, Any, List, Optional, Type
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.initializers.common_initializer import CommonInitializer, create_common_initializer
from smartcash.ui.utils.ui_logger_namespace import PREPROCESSING_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.handlers.config_handlers import ConfigHandler

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PREPROCESSING_LOGGER_NAMESPACE]

# Import components
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingConfigHandler(ConfigHandler):
    """Config handler untuk preprocessing dengan fixed implementation"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        # Implementasi sederhana untuk extract config dari UI
        config = {}
        if 'img_size_slider' in ui_components:
            config['img_size'] = ui_components['img_size_slider'].value
        if 'normalize_checkbox' in ui_components:
            config['normalize'] = ui_components['normalize_checkbox'].value
        if 'num_workers_slider' in ui_components:
            config['num_workers'] = ui_components['num_workers_slider'].value
        return {'preprocessing': config}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        preprocessing_config = config.get('preprocessing', {})
        if 'img_size_slider' in ui_components and 'img_size' in preprocessing_config:
            ui_components['img_size_slider'].value = preprocessing_config['img_size']
        if 'normalize_checkbox' in ui_components and 'normalize' in preprocessing_config:
            ui_components['normalize_checkbox'].value = preprocessing_config['normalize']
        if 'num_workers_slider' in ui_components and 'num_workers' in preprocessing_config:
            ui_components['num_workers_slider'].value = preprocessing_config['num_workers']
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk preprocessing"""
        try:
            env_manager = get_environment_manager()
            paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
            return {
                'data': {'dir': paths['data_root']},
                'preprocessing': {
                    'img_size': [640, 640], 'normalize': True, 'num_workers': 4,
                    'output_dir': paths.get('preprocessed', 'data/preprocessed')
                }
            }
        except Exception:
            return {
                'data': {'dir': 'data'},
                'preprocessing': {'img_size': [640, 640], 'normalize': True, 'num_workers': 4}
            }

class PreprocessingInitializer(CommonInitializer):
    """Initializer preprocessing terintegrasi dengan dataset/preprocessors"""
    
    def __init__(self, module_name: str = 'preprocessing', config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = 'dataset'):
        super().__init__(module_name, config_handler_class or PreprocessingConfigHandler, parent_module)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan integrasi preprocessors"""
        ui_components = create_preprocessing_main_ui(config)
        ui_components.update({
            'preprocessing_initialized': True,
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'preprocessed_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan integrasi dataset/preprocessors"""
        return setup_preprocessing_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan environment awareness"""
        # Menggunakan config handler untuk mendapatkan default config
        config_handler = self._create_config_handler()
        return config_handler.get_default_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'progress_container', 'show_for_operation', 
            'update_progress', 'complete_operation', 'error_operation', 'reset_all'
        ]

# Global instance dan public API
_preprocessing_initializer = PreprocessingInitializer()

def initialize_dataset_preprocessing_ui(env=None, config=None, parent_callbacks=None, **kwargs):
    """Factory function untuk preprocessing UI dengan parent module support"""
    return _preprocessing_initializer.initialize(env=env, config=config, parent_callbacks=parent_callbacks, **kwargs)
    
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui
"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer preprocessing yang terintegrasi dengan dataset/preprocessors tanpa duplikasi
"""

from typing import Dict, Any, List
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import PREPROCESSING_LOGGER_NAMESPACE, KNOWN_NAMESPACES

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PREPROCESSING_LOGGER_NAMESPACE]

# Import components
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingInitializer(CommonInitializer):
    """Initializer preprocessing terintegrasi dengan dataset/preprocessors"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, PREPROCESSING_LOGGER_NAMESPACE)
    
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
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'progress_container', 'show_for_operation', 
            'update_progress', 'complete_operation', 'error_operation', 'reset_all'
        ]

# Global instance dan public API
_preprocessing_initializer = PreprocessingInitializer()
initialize_dataset_preprocessing_ui = lambda env=None, config=None: _preprocessing_initializer.initialize(env=env, config=config)
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui
"""
File: smartcash/ui/pretrained_model/pretrained_initializer.py
Deskripsi: Fixed initializer dengan proper inheritance dan imports
"""

from typing import Dict, Any, List
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import PRETRAINED_MODEL_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.pretrained_model.components.ui_components import create_pretrained_main_ui
from smartcash.ui.pretrained_model.handlers.pretrained_handlers import setup_pretrained_handlers

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PRETRAINED_MODEL_LOGGER_NAMESPACE]

class PretrainedModelInitializer(CommonInitializer):
    """Initializer pretrained model terintegrasi dengan architecture yang konsisten"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, PRETRAINED_MODEL_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan integrasi pretrained model services"""
        ui_components = create_pretrained_main_ui(config)
        ui_components.update({
            'pretrained_model_initialized': True,
            'models_dir': config.get('models_dir', '/content/models'),
            'drive_models_dir': config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan integrasi pretrained model services"""
        return setup_pretrained_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan constants dari single source"""
        from smartcash.ui.pretrained_model.constants.model_constants import DEFAULT_MODELS_DIR, DEFAULT_DRIVE_MODELS_DIR
        return {
            'models_dir': DEFAULT_MODELS_DIR,
            'drive_models_dir': DEFAULT_DRIVE_MODELS_DIR
        }
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'download_sync_button', 'reset_ui_button',
            'log_output', 'status_panel', 'progress_container'
        ]

# Global instance dan public API
_pretrained_initializer = PretrainedModelInitializer()
initialize_pretrained_model_ui = lambda env=None, config=None: _pretrained_initializer.initialize(env=env, config=config)
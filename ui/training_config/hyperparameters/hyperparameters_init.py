"""
File: smartcash/ui/training_config/hyperparameters/hyperparameters_init.py
Deskripsi: Config cell untuk konfigurasi hyperparameter dengan pattern yang konsisten
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.training_config.hyperparameters.components.main_components import create_hyperparameters_form
from smartcash.ui.training_config.hyperparameters.handlers.config_extractor import extract_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_updater import update_hyperparameters_ui
from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config


class HyperparametersConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters dengan clean pattern"""
    
    def __init__(self, module_name='hyperparameters', config_filename='hyperparameters_config', config_handler_class=None,
                 parent_module: Optional[str] = 'training'):
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters config"""
        return create_hyperparameters_form(config)


class HyperparametersConfigHandler(ConfigHandler):
    """Config handler untuk hyperparameters configuration"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        return extract_hyperparameters_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        update_hyperparameters_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default hyperparameters configuration"""
        return get_default_hyperparameters_config()


def initialize_hyperparameters_config(env=None, config=None, parent_callbacks=None, **kwargs):
    """
    Factory function untuk hyperparameters config cell
    
    Args:
        env: Environment manager instance
        config: Override config values
        parent_callbacks: Callbacks untuk parent modules (training, evaluation)
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI
    """
    return create_config_cell(
        HyperparametersConfigInitializer, 
        'hyperparameters', 
        'hyperparameters_config', 
        env=env, 
        config=config, 
        config_handler_class=HyperparametersConfigHandler,
        parent_module='training',
        parent_callbacks=parent_callbacks,
        **kwargs
    )
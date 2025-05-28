"""
File: smartcash/ui/training_config/hyperparameters/hyperparameters_init.py
Deskripsi: Config cell untuk konfigurasi hyperparameter dengan pattern yang konsisten
"""

from typing import Dict, Any
from smartcash.ui.utils.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.training_config.hyperparameters.components.main_components import create_hyperparameters_form
from smartcash.ui.training_config.hyperparameters.handlers.config_extractor import extract_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_updater import update_hyperparameters_ui
from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config


class HyperparametersConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters dengan clean pattern"""
    
    def __init__(self, module_name='hyperparameters', config_filename='hyperparameters_config'):
        super().__init__(module_name, config_filename)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters config"""
        return create_hyperparameters_form(config)
    
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        return extract_hyperparameters_config(ui_components)
    
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_hyperparameters_ui(ui_components, config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default hyperparameters config"""
        return get_default_hyperparameters_config()


def initialize_hyperparameters_config(env=None, config=None, **kwargs):
    """Factory function untuk hyperparameters config cell"""
    return create_config_cell(HyperparametersConfigInitializer, 'hyperparameters', 'hyperparameters_config', env, config, **kwargs)
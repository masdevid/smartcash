"""
File: smartcash/ui/training_config/training_strategy/training_strategy_init.py
Deskripsi: Config cell untuk strategi pelatihan dengan ConfigCellInitializer yang DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.training_config.training_strategy.components.training_strategy_form import create_training_strategy_form
from smartcash.ui.training_config.training_strategy.components.training_strategy_layout import create_training_strategy_layout
from smartcash.ui.training_config.training_strategy.handlers.ui_extractor import extract_training_strategy_config
from smartcash.ui.training_config.training_strategy.handlers.ui_updater import update_training_strategy_ui
from smartcash.ui.training_config.training_strategy.handlers.defaults import get_default_training_strategy_config


class TrainingStrategyConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk strategi pelatihan yang DRY dan sederhana"""
    
    def __init__(self, module_name='training_strategy', config_filename='training_strategy_config'):
        super().__init__(module_name, config_filename)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components dengan reusable form dan layout"""
        form_components = create_training_strategy_form(config)
        return create_training_strategy_layout(form_components)
    
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI menggunakan handler yang DRY"""
        return extract_training_strategy_config(ui_components)
    
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config menggunakan handler yang DRY"""
        update_training_strategy_ui(ui_components, config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default config menggunakan handler yang DRY"""
        return get_default_training_strategy_config()


def initialize_training_strategy_config(env=None, config=None, **kwargs):
    """Factory function untuk training strategy config cell"""
    return create_config_cell(TrainingStrategyConfigInitializer, 'training_strategy', 'training_strategy_config', env, config, **kwargs)

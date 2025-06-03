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
from smartcash.common.config.manager import get_config_manager


class TrainingStrategyConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk strategi pelatihan yang DRY dan sederhana"""
    
    def __init__(self, module_name='training_strategy', config_filename='training_config'):
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
        
    def _load_config(self, config):
        """Override untuk memastikan konfigurasi dimuat dari semua file yang relevan"""
        if config:
            return config
            
        # Memuat konfigurasi dari training_config.yaml yang sudah mewarisi dari file lain
        config_manager = get_config_manager()
        training_config = config_manager.get_config(self.config_filename)
        
        # Memastikan juga memuat hyperparameters dan model terbaru
        # untuk mengatasi masalah inheritance yang tidak selalu terupdate
        hyperparameter_config = config_manager.get_config('hyperparameters')
        model_config = config_manager.get_config('model')
        
        # Jika tidak ada konfigurasi, gunakan default
        if not training_config:
            return self._get_default_config()
            
        # Menggabungkan konfigurasi jika diperlukan
        if hyperparameter_config:
            for key, value in hyperparameter_config.items():
                if key not in training_config:
                    training_config[key] = value
                    
        if model_config:
            for key, value in model_config.items():
                if key not in training_config:
                    training_config[key] = value
                    
        return training_config


def initialize_training_strategy_config(env=None, config=None, **kwargs):
    """Factory function untuk training strategy config cell"""
    return create_config_cell(TrainingStrategyConfigInitializer, 'training_strategy', 'training_config', env, config, **kwargs)

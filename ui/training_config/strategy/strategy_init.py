"""
File: smartcash/ui/training_config/strategy/strategy_init.py
Deskripsi: Config cell untuk strategi pelatihan dengan ConfigCellInitializer yang DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.utils.config_handlers import ConfigHandler
from smartcash.ui.training_config.strategy.components.ui_form import create_strategy_form
from smartcash.ui.training_config.strategy.components.ui_layout import create_strategy_layout
from smartcash.ui.training_config.strategy.handlers.config_extractor import extract_strategy_config
from smartcash.ui.training_config.strategy.handlers.config_updater import update_strategy_ui
from smartcash.ui.training_config.strategy.handlers.defaults import get_default_strategy_config
from smartcash.common.config.manager import get_config_manager


class TrainingStrategyConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk strategi pelatihan yang DRY dan sederhana"""
    
    def __init__(self, module_name='strategy', config_filename='training_config', config_handler_class=None):
        super().__init__(module_name, config_filename, config_handler_class)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components dengan reusable form dan layout"""
        form_components = create_strategy_form(config)
        return create_strategy_layout(form_components)
        
    def _load_config(self, config=None) -> Dict[str, Any]:
        """Override untuk memastikan konfigurasi dimuat dari semua file yang relevan"""
        if config:
            return config
            
        try:
            # Memuat konfigurasi dari training_config.yaml yang sudah mewarisi dari file lain
            config_manager = get_config_manager()
            training_config = config_manager.get_config(self.config_filename)
            
            # Memastikan juga memuat hyperparameters dan model terbaru
            # untuk mengatasi masalah inheritance yang tidak selalu terupdate
            hyperparameter_config = config_manager.get_config('hyperparameters')
            model_config = config_manager.get_config('model')
            
            # Jika tidak ada konfigurasi, gunakan default
            if not training_config:
                if self.config_handler:
                    training_config = self.config_handler.get_default_config()
                else:
                    training_config = self._get_default_config()
            
            # Resolve base configs terlebih dahulu
            training_config = self._resolve_base_configs(training_config)
                
            # Menggabungkan konfigurasi jika diperlukan
            if hyperparameter_config:
                hyperparameter_config = self._resolve_base_configs(hyperparameter_config)
                for key, value in hyperparameter_config.items():
                    if key not in training_config:
                        training_config[key] = value
                        
            if model_config:
                model_config = self._resolve_base_configs(model_config)
                for key, value in model_config.items():
                    if key not in training_config:
                        training_config[key] = value
                        
            return training_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading config: {str(e)}")
            if self.config_handler:
                return self.config_handler.get_default_config()
            else:
                return self._get_default_config()


class StrategyConfigHandler(ConfigHandler):
    """Config handler untuk training strategy configuration"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        return extract_strategy_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        update_strategy_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default strategy configuration"""
        return get_default_strategy_config()


def initialize_strategy_config(env=None, config=None, **kwargs):
    """
    Factory function untuk training strategy config cell
    
    Args:
        env: Environment manager instance
        config: Override config values
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI
    """
    return create_config_cell(
        TrainingStrategyConfigInitializer, 
        'strategy', 
        'training_config', 
        env, 
        config, 
        config_handler_class=StrategyConfigHandler, 
        **kwargs
    )

"""
File: smartcash/ui/hyperparameters/hyperparameters_init.py
Deskripsi: Initializer untuk hyperparameters configuration menggunakan hyperparameters_config.yaml
"""

import traceback
import sys
from typing import Dict, Any, Optional

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.common.logger import get_logger

logger = get_logger(__name__)
MODULE_NAME = 'hyperparameters'
MODULE_CONFIG = f"{MODULE_NAME}_config"


class HyperparametersInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters configuration"""
    
    def __init__(self, module_name: str = MODULE_NAME, config_filename: str = MODULE_CONFIG, 
                 config_handler_class=None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            from .handlers.config_handler import HyperparametersConfigHandler
            config_handler_class = HyperparametersConfigHandler
            
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters configuration"""
        try:
            from .components.ui_form import create_hyperparameters_form
            from .components.ui_layout import create_hyperparameters_layout
            
            # Pastikan config memiliki struktur yang benar
            if not isinstance(config, dict):
                config = {}
                
            # Buat form components
            form_components = create_hyperparameters_form(config)
            
            # Buat layout dengan form components
            return create_hyperparameters_layout(form_components)
            
        except Exception as e:
            return self.handle_ui_exception(e, context="UI hyperparameters configuration")
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> None:
        """Setup custom handlers untuk hyperparameters changes"""
        try:
            # Add any custom handlers for hyperparameters here
            pass
        except Exception as e:
            logger.warning(f"Gagal setup custom handlers: {str(e)}")


def initialize_hyperparameters_config(env=None, config=None, parent_callbacks=None, **kwargs) -> Any:
    """
    Factory function untuk hyperparameters config cell dengan parent integration.
    
    Args:
        env: Environment manager instance
        config: Override config values
        parent_callbacks: Dict callbacks untuk parent modules
        **kwargs: Additional arguments
        
    Returns:
        UI components dengan config
    """
    try:
        # Inisialisasi dengan config handler
        initializer = HyperparametersInitializer()
        
        # Buat config cell
        return create_config_cell(
            initializer=initializer,
            env=env,
            config=config,
            parent_callbacks=parent_callbacks,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Gagal menginisialisasi hyperparameters config: {str(e)}")
        logger.debug(traceback.format_exc())
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        return create_fallback_ui(
            title="Gagal Memuat Konfigurasi Hyperparameters",
            error=str(e),
            help_text="Silakan periksa log untuk detail lebih lanjut."
        )
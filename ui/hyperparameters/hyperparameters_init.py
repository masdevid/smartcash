"""
File: smartcash/ui/hyperparameters/hyperparameters_init.py
Deskripsi: Main initializer untuk hyperparameters config cell dengan clean pattern dan config inheritance
"""

import sys
import traceback
from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.hyperparameters.handlers.config_handler import HyperparametersConfigHandler
from smartcash.ui.hyperparameters.components.ui_form import create_hyperparameters_form
from smartcash.ui.hyperparameters.components.ui_layout import create_hyperparameters_layout


class HyperparametersConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters dengan inheritance dari config_cell_initializer"""
    
    def __init__(self, module_name='hyperparameters', config_filename='hyperparameters_config', 
                 config_handler_class=None, parent_module: Optional[str] = None):
        super().__init__(module_name, config_filename, config_handler_class or HyperparametersConfigHandler, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters config dengan form dan layout"""
        try:
            # Create form components
            form_components = create_hyperparameters_form(config)
            
            # Create layout dengan form components
            layout_components = create_hyperparameters_layout(form_components)
            
            return layout_components
            
        except Exception as e:
            error_msg = f"Gagal membuat UI hyperparameters: {str(e)}"
            from smartcash.common.logger import get_logger
            logger = get_logger(__name__)
            logger.exception(error_msg, exc_info=True)
            
            # Dapatkan traceback dari sys.exc_info()
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb)) if exc_tb else ""
            
            from smartcash.ui.utils.fallback_utils import create_fallback_ui, FallbackConfig
            return create_fallback_ui(
                error_message=error_msg,
                exc_info=sys.exc_info(),
                config=FallbackConfig(
                    title="⚠️ Error Hyperparameters Configuration",
                    module_name='hyperparameters',
                    traceback=tb_text
                )
            )
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup custom handlers untuk hyperparameters jika diperlukan"""
        # Custom logic untuk hyperparameters bisa ditambahkan di sini jika diperlukan
        pass


def initialize_hyperparameters_config(env=None, config=None, parent_callbacks=None, **kwargs):
    """
    Factory function untuk hyperparameters config cell dengan clean initialization
    
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
        parent_module=None,  # Hyperparameters adalah modul independen
        parent_callbacks=parent_callbacks,
        **kwargs
    )


# Backward compatibility untuk existing imports
def create_hyperparameters_config_cell(env=None, config=None, **kwargs):
    """Backward compatibility function"""
    return initialize_hyperparameters_config(env=env, config=config, **kwargs)
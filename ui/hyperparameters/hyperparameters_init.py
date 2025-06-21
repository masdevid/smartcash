"""
File: smartcash/ui/hyperparameters/hyperparameters_init.py
Deskripsi: Initializer untuk hyperparameters configuration menggunakan hyperparameters_config.yaml
"""

import traceback
import sys
from typing import Dict, Any, Optional, Type

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.common.logger import get_logger

logger = get_logger(__name__)
MODULE_NAME = 'hyperparameters'
MODULE_CONFIG = f"{MODULE_NAME}_config"


class HyperparametersInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters configuration"""
    
    def __init__(self, module_name: str = MODULE_NAME, config_filename: str = MODULE_CONFIG, 
                 config_handler_class: Optional[Type] = None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            from .handlers.config_handler import HyperparametersConfigHandler
            config_handler_class = HyperparametersConfigHandler
            
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters configuration"""
        try:
            from .components.ui_form import create_hyperparameters_form
            from .components.ui_layout import create_hyperparameters_layout
            from ..utils.fallback_utils import create_fallback_ui
            
            # Pastikan config memiliki struktur yang benar
            if not isinstance(config, dict):
                config = {}
            
            # Buat form components
            form_components = create_hyperparameters_form(config)
            
            # Buat layout dengan form components
            layout_components = create_hyperparameters_layout(form_components)
            
            # Gabungkan semua komponen
            return {
                **layout_components,
                'form_components': form_components
            }
            
        except Exception as e:
            logger.error(f"❌ Gagal membuat UI hyperparameters: {str(e)}")
            logger.debug(traceback.format_exc())
            return create_fallback_ui(
                error_message=str(e),
                module_name=self.module_name,
                ui_components={"help_text": "Gagal memuat konfigurasi hyperparameters. Silakan periksa log untuk detail lebih lanjut."}
            )
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> None:
        """Setup custom handlers untuk hyperparameters changes"""
        try:
            # Setup change handlers untuk form components
            form_components = ui_components.get('form_components', {})
            
            # Contoh: Handle perubahan pada learning rate
            if 'learning_rate' in form_components:
                def on_learning_rate_change(change):
                    logger.info(f"Learning rate berubah menjadi: {change['new']}")
                    
                form_components['learning_rate'].observe(on_learning_rate_change, names='value')
                
            # Tambahkan handler lain sesuai kebutuhan
            
        except Exception as e:
            logger.warning(f"⚠️ Gagal setup custom handlers: {str(e)}")
            logger.debug(traceback.format_exc())


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
        return create_config_cell(
            initializer_class=type(initializer),
            module_name=initializer.module_name,
            config_filename=initializer.config_filename,
            env=env,
            config=config,
            parent_callbacks=parent_callbacks,
            **kwargs
        )
    except Exception as e:
        logger.error(f"❌ Gagal menginisialisasi hyperparameters config: {str(e)}")
        logger.debug(traceback.format_exc())
        from smartcash.ui.utils.fallback_ui import create_fallback_ui
        return create_fallback_ui(
            error_message=str(e),
            module_name=MODULE_NAME,
            ui_components={"help_text": "Gagal menginisialisasi konfigurasi hyperparameters. Silakan periksa log untuk detail lebih lanjut."}
        )
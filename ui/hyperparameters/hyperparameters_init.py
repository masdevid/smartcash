# File: smartcash/ui/hyperparameters/hyperparameters_init.py
# Deskripsi: Updated initializer untuk hyperparameters configuration yang disederhanakan

import sys
import traceback
from typing import Dict, Any, Optional, Type

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.common.logger import get_logger

logger = get_logger(__name__)
MODULE_NAME = 'hyperparameters'
MODULE_CONFIG = f"{MODULE_NAME}_config"


class HyperparametersInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters configuration yang disederhanakan"""
    
    def __init__(self, module_name: str = MODULE_NAME, config_filename: str = MODULE_CONFIG, 
                 config_handler_class: Optional[Type] = None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            from .handlers.config_handler import HyperparametersConfigHandler
            config_handler_class = HyperparametersConfigHandler
            
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components responsive dengan color-coded groups dan fallback handling"""
        try:
            from .components.ui_form import create_hyperparameters_form
            from .components.ui_layout import create_hyperparameters_layout
            
            # Pastikan config valid
            if not isinstance(config, dict):
                config = {}
            
            # Buat form components dengan enhanced styling
            form_components = create_hyperparameters_form(config)
            
            # Jika form_components adalah fallback UI (mengandung 'main_layout'), langsung kembalikan
            if 'main_layout' in form_components:
                logger.warning("⚠️ Menggunakan fallback UI untuk hyperparameters")
                return form_components
            
            # Buat responsive layout dengan color-coded groups
            try:
                layout_components = create_hyperparameters_layout(form_components)
                logger.info("✅ UI hyperparameters responsive berhasil dibuat")
                
                return {
                    **layout_components,
                    'form_components': form_components,
                    'main_layout': layout_components.get('main_layout')
                }
                
            except Exception as layout_error:
                logger.error(f"❌ Gagal membuat layout hyperparameters: {str(layout_error)}", exc_info=True)
                return form_components  # Kembalikan form components saja jika layout gagal
            
            
        except Exception as e:
            logger.error(f"❌ Gagal membuat UI hyperparameters: {str(e)}", exc_info=True)
            # Coba buat fallback UI
            try:
                from .components.ui_form import _create_fallback_ui
                return _create_fallback_ui(
                    f"Gagal memuat UI hyperparameters: {str(e)}",
                    exc_info=sys.exc_info(),
                    show_traceback=True,
                    retry_callback=lambda: self._create_config_ui(config, env, **kwargs)
                )
            except Exception as fallback_error:
                logger.critical(f"❌ Gagal membuat fallback UI: {str(fallback_error)}")
                raise RuntimeError("Gagal memuat UI dan fallback UI tidak tersedia") from e
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> None:
        """Setup validation dan change handlers"""
        try:
            from .utils.form_validation import validate_hyperparameters_form
            
            form_components = ui_components.get('form_components', {})
            
            # Setup real-time validation pada key parameters
            key_widgets = ['learning_rate', 'batch_size', 'epochs']
            for widget_name in key_widgets:
                if widget_name in form_components:
                    def create_validator(name):
                        def validate_change(change):
                            is_valid, errors = validate_hyperparameters_form(form_components)
                            if not is_valid:
                                logger.warning(f"⚠️ Validation warning untuk {name}: {errors[0] if errors else 'Unknown'}")
                        return validate_change
                    
                    form_components[widget_name].observe(create_validator(widget_name), names='value')
                    
        except Exception as e:
            logger.warning(f"⚠️ Gagal setup custom handlers: {str(e)}")


def initialize_hyperparameters_config(env=None, config=None, parent_callbacks=None, **kwargs):
    """Factory function untuk hyperparameters config cell yang disederhanakan"""
    try:
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
        
        # Simple fallback
        import ipywidgets as widgets
        return widgets.HTML(value=f"<div style='color: red;'>❌ Init Error: {str(e)}</div>")
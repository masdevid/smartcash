"""
File: smartcash/ui/hyperparameters/hyperparameters_init.py
Deskripsi: Main initializer untuk hyperparameters config cell dengan clean pattern dan config inheritance
"""

import sys
import traceback
from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.hyperparameters.handlers.config_handler import HyperparametersConfigHandler
from smartcash.ui.hyperparameters.components.ui_form import create_hyperparameters_form
from smartcash.common.logger import get_logger


class HyperparametersConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters dengan inheritance dari config_cell_initializer"""
    
    def __init__(self, module_name='hyperparameters', config_filename='hyperparameters_config', 
                 config_handler_class=None, parent_module: Optional[str] = None):
        super().__init__(module_name, config_filename, config_handler_class or HyperparametersConfigHandler, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters config dengan form dan layout"""
        try:
            from .components.ui_form import create_hyperparameters_form
            from .components.ui_layout import create_hyperparameters_layout
            import traceback
            import ipywidgets as widgets
            
            # Debug: Log config yang diterima
            self.logger.debug(f"Membuat UI dengan config: {config}")
            
            # Buat form components
            try:
                form_components = create_hyperparameters_form(config)
                self.logger.debug("Form components berhasil dibuat")
            except Exception as e:
                self.logger.error(f"Gagal membuat form components: {str(e)}\n{traceback.format_exc()}")
                raise ValueError(f"Gagal membuat form components: {str(e)}") from e
            
            # Debug: Tampilkan semua kunci yang tersedia di form_components
            available_components = list(form_components.keys())
            self.logger.debug(f"Komponen yang tersedia di form_components: {available_components}")
            
            # Pastikan komponen yang diperlukan ada di form_components
            required_form_components = [
                'epochs_slider', 'batch_size_slider', 'learning_rate_slider',
                'optimizer_dropdown', 'weight_decay_slider', 'scheduler_dropdown',
                'warmup_epochs_slider', 'box_loss_gain_slider', 'cls_loss_gain_slider',
                'obj_loss_gain_slider', 'early_stopping_checkbox', 'patience_slider',
                'save_best_checkbox', 'checkpoint_metric_dropdown', 'summary_cards',
                'status_panel', 'button_container', 'save_button', 'reset_button'
            ]
            
            missing_components = [comp for comp in required_form_components if comp not in form_components]
            if missing_components:
                error_msg = (
                    f"Komponen form yang hilang: {missing_components}\n"
                    f"Komponen yang tersedia: {available_components}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Buat layout dengan form components
            try:
                layout_components = create_hyperparameters_layout(form_components)
                self.logger.debug("Layout components berhasil dibuat")
            except Exception as e:
                self.logger.error(f"Gagal membuat layout components: {str(e)}\n{traceback.format_exc()}")
                raise ValueError(f"Gagal membuat layout components: {str(e)}") from e
            
            # Debug: Tampilkan semua kunci yang tersedia di layout_components
            available_layout = list(layout_components.keys())
            self.logger.debug(f"Komponen yang tersedia di layout_components: {available_layout}")
            
            # Pastikan komponen yang diperlukan ada di layout_components
            required_layout_components = ['form', 'save_button', 'reset_button']
            missing_layout = [comp for comp in required_layout_components if comp not in layout_components]
            if missing_layout:
                error_msg = (
                    f"Komponen layout yang hilang: {missing_layout}\n"
                    f"Komponen yang tersedia: {available_layout}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Pastikan form adalah widget yang valid
            if not isinstance(layout_components['form'], widgets.Widget):
                error_msg = f"Form harus berupa instance widgets.Widget, tapi mendapat: {type(layout_components['form'])}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Return komponen yang diperlukan
            result = {
                'form': layout_components['form'],
                'save_button': layout_components['save_button'],
                'reset_button': layout_components['reset_button'],
                'container': layout_components['form']
            }
            
            self.logger.debug("UI components berhasil dibuat")
            return result
                
        except Exception as e:
            self.logger.error(f"Error di _create_config_ui: {str(e)}\n{traceback.format_exc()}")
            return self.handle_ui_exception(e, context="UI hyperparameters")
    
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
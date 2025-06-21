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
            
            # Buat form components
            form_components = create_hyperparameters_form(config)
            
            # Pastikan komponen yang diperlukan ada di form_components
            required_form_components = [
                'epochs_slider', 'batch_size_slider', 'learning_rate_slider',
                'optimizer_dropdown', 'weight_decay_slider', 'scheduler_dropdown',
                'warmup_epochs_slider', 'box_loss_gain_slider', 'cls_loss_gain_slider',
                'obj_loss_gain_slider', 'early_stopping_checkbox', 'patience_slider',
                'save_best_checkbox', 'checkpoint_metric_dropdown', 'summary_cards',
                'status_panel', 'button_container'
            ]
            
            for comp in required_form_components:
                if comp not in form_components:
                    raise ValueError(f"Komponen form '{comp}' tidak ditemukan")
            
            # Buat layout dengan form components
            layout_components = create_hyperparameters_layout(form_components)
            
            # Pastikan komponen yang diperlukan ada di layout_components
            required_layout_components = ['form', 'save_button', 'reset_button']
            for comp in required_layout_components:
                if comp not in layout_components:
                    raise ValueError(f"Komponen layout '{comp}' tidak ditemukan")
            
            # Pastikan form adalah widget yang valid
            if not isinstance(layout_components['form'], widgets.Widget):
                raise ValueError("Form harus berupa instance widgets.Widget yang valid")
            
            # Return komponen yang diperlukan
            return {
                'form': layout_components['form'],
                'save_button': layout_components['save_button'],
                'reset_button': layout_components['reset_button'],
                'container': layout_components['form']
            }
            
        except Exception as e:
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
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
            
            # Pastikan config memiliki struktur yang benar
            if not isinstance(config, dict):
                config = {}
                
            # Inisialisasi section yang diperlukan
            if 'training' not in config:
                config['training'] = {}
            if 'optimizer' not in config:
                config['optimizer'] = {}
            if 'scheduler' not in config:
                config['scheduler'] = {}
            if 'loss' not in config:
                config['loss'] = {}
            if 'early_stopping' not in config:
                config['early_stopping'] = {}
            if 'checkpoint' not in config:
                config['checkpoint'] = {}
            
            # Debug: Log config yang diterima
            self.logger.debug(f"Membuat UI dengan config: {config}")
            
            # Buat form components
            try:
                form_components = create_hyperparameters_form(config)
                self.logger.debug("Form components berhasil dibuat")
            except Exception as e:
                error_msg = f"Gagal membuat form components: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return self.handle_ui_exception(ValueError(error_msg), context="Membuat form components")
            
            # Pastikan form_components adalah dictionary
            if not isinstance(form_components, dict):
                error_msg = f"form_components harus berupa dictionary, tapi mendapat: {type(form_components)}"
                self.logger.error(error_msg)
                return self.handle_ui_exception(ValueError(error_msg), context="Validasi form components")
            
            # Buat layout dengan form components
            try:
                layout_components = create_hyperparameters_layout(form_components)
                self.logger.debug("Layout components berhasil dibuat")
            except Exception as e:
                error_msg = f"Gagal membuat layout components: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                return self.handle_ui_exception(ValueError(error_msg), context="Membuat layout components")
            
            # Pastikan layout_components adalah dictionary
            if not isinstance(layout_components, dict):
                error_msg = f"layout_components harus berupa dictionary, tapi mendapat: {type(layout_components)}"
                self.logger.error(error_msg)
                return self.handle_ui_exception(ValueError(error_msg), context="Validasi layout components")
            
            # Pastikan komponen yang diperlukan ada
            required_components = ['form', 'save_button', 'reset_button']
            missing_components = [comp for comp in required_components if comp not in layout_components]
            
            if missing_components:
                error_msg = f"Komponen UI yang diperlukan tidak ditemukan: {missing_components}"
                self.logger.error(error_msg)
                return self.handle_ui_exception(ValueError(error_msg), context="Validasi komponen UI")
            
            # Pastikan form adalah widget yang valid
            form = layout_components['form']
            if not isinstance(form, widgets.Widget):
                error_msg = f"Form harus berupa instance widgets.Widget, tapi mendapat: {type(form)}"
                self.logger.error(error_msg)
                return self.handle_ui_exception(ValueError(error_msg), context="Validasi tipe widget")
            
            # Return komponen yang diperlukan
            result = {
                'form': form,
                'save_button': layout_components['save_button'],
                'reset_button': layout_components['reset_button'],
                'container': form,
                'config_handler': self.config_handler
            }
            
            # Setup callback untuk update summary
            self._setup_summary_update_callback(result)
            
            self.logger.debug("UI components berhasil dibuat")
            return result
                
        except Exception as e:
            self.logger.error(f"Error di _create_config_ui: {str(e)}\n{traceback.format_exc()}")
            return self.handle_ui_exception(e, context="Membuat UI hyperparameters")
    
    def _create_ui_with_config(self, config, env=None, **kwargs):
        """Membuat UI dengan konfigurasi yang diberikan"""
        from .hyperparameters_init import create_hyperparameters_config_cell
        config_cell = create_hyperparameters_config_cell(env=env, config=config, **kwargs)
        
        # Pastikan config_cell memiliki method get_ui()
        if not hasattr(config_cell, 'get_ui'):
            if hasattr(config_cell, 'main_container'):
                config_cell.get_ui = lambda: config_cell.main_container
            elif isinstance(config_cell, dict) and 'main_container' in config_cell:
                config_cell.get_ui = lambda: config_cell['main_container']
            else:
                config_cell.get_ui = lambda: config_cell
        
        return config_cell
    
    def initialize(self, env=None, config=None, **kwargs):
        """Optimized initialization dengan proper error handling"""
        try:
            # Panggil parent initialize
            result = super().initialize(env=env, config=config, **kwargs)
            
            # Buat UI dengan konfigurasi yang diberikan
            config_cell = self._create_ui_with_config(config, env, **kwargs)
            
            # Panggil get_ui() untuk kompatibilitas dengan test
            config_cell.get_ui()
            
            return config_cell
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameters initialize: {str(e)}", exc_info=True)
            # Fallback ke parent initialize dengan default config
            fallback_config = self.config_handler.get_default_config()
            fallback_result = super().initialize(env=env, config=fallback_config, **kwargs)
            
            # Pastikan fallback result punya get_ui()
            if not hasattr(fallback_result, 'get_ui'):
                fallback_result.get_ui = lambda: fallback_result
                
            return fallback_result
    
    def _setup_summary_update_callback(self, ui_components: Dict[str, Any]) -> None:
        """Setup callback untuk update summary card otomatis"""
        try:
            from ipywidgets import Widget
            
            def on_value_change(change):
                """Callback untuk update summary saat ada perubahan nilai widget"""
                try:
                    config_handler = ui_components.get('config_handler')
                    if config_handler and hasattr(config_handler, 'extract_config'):
                        current_config = config_handler.extract_config(ui_components)
                        if 'summary_cards' in ui_components and hasattr(config_handler, 'update_ui_from_config'):
                            config_handler.update_ui_from_config(ui_components, current_config)
                except Exception as e:
                    self.logger.warning(f"Gagal update summary: {str(e)}")
            
            # Daftarkan callback untuk semua widget yang memiliki value attribute
            for name, widget in ui_components.items():
                if isinstance(widget, Widget) and hasattr(widget, 'observe') and hasattr(widget, 'value'):
                    widget.observe(on_value_change, names='value')
                    
        except Exception as e:
            self.logger.warning(f"Gagal setup summary callback: {str(e)}")
    
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
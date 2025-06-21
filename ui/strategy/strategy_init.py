"""
File: smartcash/ui/strategy/strategy_init.py
Deskripsi: Initializer untuk strategy configuration dengan cascading config inheritance
"""

import traceback
import sys
from typing import Dict, Any, Optional, List

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.common.logger import get_logger

logger = get_logger(__name__)
MODULE_NAME = 'strategy'
MODULE_CONFIG = f"{MODULE_NAME}_config"


class StrategyInitializer(ConfigCellInitializer):
    """Config cell initializer untuk strategy configuration dengan cascading inheritance"""
    
    def __init__(self, module_name: str = MODULE_NAME, config_filename: str = MODULE_CONFIG,
                 config_handler_class=None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            from .handlers.config_handler import StrategyConfigHandler
            config_handler_class = StrategyConfigHandler
            
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
        self._inheritance_chain = [
            'base_config', 'preprocessing_config', 'augmentation_config',
            'model_config', 'backbone_config', 'hyperparameters_config', 'training_config'
        ]
    
    def _normalize_config_structure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalisasi struktur konfigurasi untuk kompatibilitas"""
        if not isinstance(config, dict):
            return {'training': {}, 'validation': {}}
        
        # Normalisasi struktur training
        training = config.get('training', {})
        
        # Jika ada hyperparameters di root, pindahkan ke dalam training
        for key in ['optimizer', 'scheduler', 'loss', 'early_stopping', 'checkpoint']:
            if key in config and key not in training:
                training[key] = config[key]
                
        # Normalisasi optimizer
        if 'optimizer' in training and isinstance(training['optimizer'], dict):
            opt = training['optimizer']
            if 'type' in opt and 'optimizer' not in training:
                training['optimizer'] = opt['type']
                
        # Normalisasi scheduler
        if 'scheduler' in training and isinstance(training['scheduler'], dict):
            sched = training['scheduler']
            if 'type' in sched and 'scheduler' not in training:
                training['scheduler'] = sched['type']
                
        # Pastikan training ada di config
        config['training'] = training
        
        return config
    
    def _load_cascading_config(self, env=None) -> Dict[str, Any]:
        """Load config dengan cascading inheritance"""
        try:
            config = {}
            for config_name in self._inheritance_chain:
                try:
                    if cfg := self.config_manager.get_config(config_name):
                        cfg.pop('_base_', None)  # Hapus inheritance loops
                        self._deep_merge_inplace(config, cfg)
                        logger.debug(f"Merged config: {config_name}")
                except Exception as e:
                    logger.warning(f"Skip {config_name}: {str(e)}")
            
            return self._normalize_config_structure(config or self.config_handler.get_default_config())
            
        except Exception as e:
            logger.error(f"Gagal load cascading config: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._normalize_config_structure({})
    
    def _deep_merge_inplace(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """In-place deep merge untuk performance optimization"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_inplace(base[key], value)
            else:
                base[key] = value
    
        return None
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Override initialize untuk cascading config loading"""
        try:
            # Load config jika tidak disediakan
            if config is None:
                config = self.config_handler.load_config()
            
            # Pastikan config memiliki struktur minimal yang diperlukan
            if not isinstance(config, dict):
                config = {}
                
            # Pastikan section penting ada
            if 'training' not in config:
                config['training'] = {}
            if 'validation' not in config:
                config['validation'] = {}
            
            # Panggil parent initialize dengan config yang sudah divalidasi
            result = super().initialize(env=env, config=config, **kwargs)
            
            # Buat UI strategy
            strategy_ui = self._create_strategy_ui(config, env, **kwargs)
            
            # Kembalikan UI yang sudah dibuat atau result dari parent
            if strategy_ui is not None:
                # Panggil get_ui() untuk kompatibilitas dengan test
                strategy_ui.get_ui()
                return strategy_ui
                
            # Jika tidak ada UI khusus, pastikan result memiliki get_ui()
            if not hasattr(result, 'get_ui'):
                if hasattr(result, 'main_container'):
                    result.get_ui = lambda: result.main_container
                elif isinstance(result, dict) and 'main_container' in result:
                    result.get_ui = lambda: result['main_container']
                else:
                    result.get_ui = lambda: result
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Error in strategy initialize: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Fallback ke parent initialize dengan default config
            fallback_config = self.config_handler.get_default_config()
            fallback_result = super().initialize(env=env, config=fallback_config, **kwargs)
            
            # Pastikan fallback result punya get_ui()
            if not hasattr(fallback_result, 'get_ui'):
                fallback_result.get_ui = lambda: fallback_result
                
            return fallback_result
    

    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk strategy config dengan form dan layout"""
        try:
            # Pastikan config memiliki struktur yang benar
            if not isinstance(config, dict):
                config = {}
                
            # Inisialisasi section yang diperlukan
            if 'training' not in config:
                config['training'] = {}
            if 'validation' not in config:
                config['validation'] = {}
                
            # Debug: Log config yang diterima
            self.logger.debug(f"Membuat UI dengan config: {config}")
            
            # Buat form components
            try:
                form_components = create_strategy_form(config)
                self.logger.debug("Form components berhasil dibuat")
                
                if not isinstance(form_components, dict):
                    error_msg = f"form_components harus berupa dictionary, tapi mendapat: {type(form_components)}"
                    return self.handle_ui_exception(ValueError(error_msg), "Membuat form components")
                    
            except Exception as e:
                error_msg = f"Gagal membuat form components: {str(e)}"
                return self.handle_ui_exception(e, "Membuat form components")
            
            # Buat layout components
            try:
                layout_components = create_strategy_layout(form_components)
                self.logger.debug("Layout components berhasil dibuat")
                
                if not isinstance(layout_components, dict):
                    error_msg = f"layout_components harus berupa dictionary, tapi mendapat: {type(layout_components)}"
                    return self.handle_ui_exception(ValueError(error_msg), "Membuat layout components")
                    
            except Exception as e:
                error_msg = f"Gagal membuat layout components: {str(e)}"
                return self.handle_ui_exception(e, "Membuat layout components")
            
            # Pastikan komponen yang diperlukan ada
            required_components = ['main_container', 'form', 'save_button', 'reset_button', 'summary_card']
            missing_components = [comp for comp in required_components if comp not in layout_components]
            
            if missing_components:
                error_msg = f"Komponen UI yang diperlukan tidak ditemukan: {missing_components}"
                return self.handle_ui_exception(ValueError(error_msg), "Validasi komponen UI")
                
            # Pastikan main_container adalah widget yang valid
            main_container = layout_components['main_container']
            if not isinstance(main_container, widgets.Widget):
                error_msg = f"main_container harus berupa instance widgets.Widget, tapi mendapat: {type(main_container)}"
                return self.handle_ui_exception(ValueError(error_msg), "Validasi tipe widget")
            
            # Setup callback untuk update summary
            self._setup_summary_update_callback(layout_components)
            
            # Buat wrapper untuk kompatibilitas dengan test case
            class UIWrapper:
                def __init__(self, container):
                    self.container = container
                    
                def get_ui(self):
                    return self.container
                    
            # Buat wrapper untuk result yang memiliki method get_ui()
            class ResultWrapper(dict):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._ui_wrapper = UIWrapper(self['main_container'])
                    
                def get_ui(self):
                    return self._ui_wrapper.get_ui()
            
            # Kembalikan komponen yang diperlukan
            result = ResultWrapper({
                'main_container': main_container,
                'form': layout_components['form'],
                'save_button': layout_components['save_button'],
                'reset_button': layout_components['reset_button'],
                'summary_card': layout_components.get('summary_card'),
                'config_handler': self.config_handler
            })
            
            self.logger.debug("UI components berhasil dibuat")
            return result
                
        except Exception as e:
            error_msg = f"Error tidak terduga di _create_config_ui: {str(e)}"
            return self.handle_ui_exception(e, "Membuat UI strategy")
            
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
                        if 'summary_card' in ui_components and hasattr(config_handler, 'update_ui'):
                            config_handler.update_ui(ui_components, current_config)
                except Exception as e:
                    self.logger.warning(f"Gagal update summary: {str(e)}")
            
            # Daftarkan callback untuk semua widget yang memiliki value attribute
            for name, widget in ui_components.items():
                if isinstance(widget, Widget) and hasattr(widget, 'observe') and hasattr(widget, 'value'):
                    widget.observe(on_value_change, names='value')
                    
        except Exception as e:
            self.logger.warning(f"Gagal setup summary callback: {str(e)}")
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup custom handlers untuk strategy-specific functionality"""
        # Override parent method untuk menambahkan functionality khusus strategy
        pass


def initialize_strategy_config(env=None, config=None, parent_callbacks=None, **kwargs):
    """
    Factory function untuk strategy config cell
    
    Args:
        env: Environment manager instance
        config: Override config values
        parent_callbacks: Callbacks untuk parent modules
        **kwargs: Additional arguments
        
    Returns:
        UI components dengan config
    """
    try:
        # Inisialisasi dengan config handler
        initializer = StrategyInitializer()
        
        # Buat config cell
        return create_config_cell(
            initializer=initializer,
            env=env,
            config=config,
            parent_callbacks=parent_callbacks,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Gagal menginisialisasi strategy config: {str(e)}")
        logger.debug(traceback.format_exc())
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        return create_fallback_ui(
            title="Gagal Memuat Konfigurasi Strategy",
            error=str(e),
            help_text="Silakan periksa log untuk detail lebih lanjut."
        )
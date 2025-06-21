"""
File: smartcash/ui/strategy/strategy_init.py
Deskripsi: Main initializer untuk strategy config cell dengan cascading inheritance support
"""

import traceback
import sys
from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.strategy.handlers.config_handler import StrategyConfigHandler
from smartcash.ui.strategy.components.ui_form import create_strategy_form
from smartcash.ui.strategy.components.ui_layout import create_strategy_layout, update_summary_card
from smartcash.ui.utils.fallback_utils import show_status_safe, create_fallback_ui
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class StrategyInitializer(ConfigCellInitializer):
    """Strategy config cell initializer dengan cascading inheritance support"""
    
    def __init__(self, module_name='strategy', config_filename='training_config', 
                 config_handler_class=None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            config_handler_class = StrategyConfigHandler
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
        self.config_manager = get_config_manager()
    
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
                
        # Pastikan ada section yang diperlukan
        config['training'] = training
        if 'validation' not in config:
            config['validation'] = {}
            
        return config
            
    def _load_cascading_config(self) -> Dict[str, Any]:
        """Load config dengan cascading inheritance sesuai urutan yang benar"""
        try:
            # Urutan inheritance: base -> preprocessing -> augmentation -> model -> backbone -> hyperparameters -> training
            inheritance_chain = [
                'base_config',
                'preprocessing_config', 
                'augmentation_config',
                'model_config',
                'backbone_config',
                'hyperparameters_config',
                'training_config'
            ]
            
            # Merge configs dalam urutan inheritance
            merged_config = {}
            for config_name in inheritance_chain:
                try:
                    config = self.config_manager.get_config(config_name)
                    if config and isinstance(config, dict):
                        # Remove _base_ untuk mencegah recursive inheritance
                        config.pop('_base_', None)
                        # Deep merge configs
                        merged_config = self._deep_merge_configs(merged_config, config)
                except Exception:
                    continue  # Skip config yang error
            
            # Normalisasi struktur konfigurasi
            merged_config = self._normalize_config_structure(merged_config)
            
            # Fallback ke defaults jika tidak ada config
            if not merged_config or 'training' not in merged_config:
                default_config = self.config_handler.get_default_config()
                if not default_config:
                    default_config = {}
                
                # Pastikan minimal ada konfigurasi dasar yang valid
                default_config = self._normalize_config_structure(default_config)
                return default_config
            
            return merged_config
            
        except Exception:
            # Kembalikan config kosong yang valid
            return self._normalize_config_structure({})
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge dua config dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Override initialize untuk cascading config loading"""
        try:
            # Load atau merge config
            if not config:
                final_config = self._load_cascading_config()
            else:
                base_config = self._load_cascading_config()
                final_config = self._deep_merge_configs(base_config, config)
            
            # Pastikan config memiliki struktur minimal yang diperlukan
            if not isinstance(final_config, dict):
                final_config = {}
                
            # Pastikan section penting ada
            if 'training' not in final_config:
                final_config['training'] = {}
            if 'validation' not in final_config:
                final_config['validation'] = {}
            
            # Call parent initialize dengan config yang sudah divalidasi
            return super().initialize(env=env, config=final_config, **kwargs)
            
        except Exception as e:
            error_msg = f"❌ Error in strategy initialize: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Fallback ke parent initialize dengan default config
            return super().initialize(env=env, config=self.config_handler.get_default_config(), **kwargs)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk strategy config"""
        try:
            from .components.ui_form import create_strategy_form
            from .components.ui_layout import create_strategy_layout, update_summary_card
            import ipywidgets as widgets
            
            # Debug: Log config yang diterima
            self.logger.debug(f"Membuat UI strategy dengan config: {config}")
            
            # Buat form components
            try:
                form_components = create_strategy_form(config)
                self.logger.debug("Form components berhasil dibuat")
            except Exception as e:
                self.logger.error(f"Gagal membuat form components: {str(e)}\n{traceback.format_exc()}")
                raise ValueError(f"Gagal membuat form components: {str(e)}") from e
            
            # Debug: Tampilkan semua kunci yang tersedia di form_components
            available_components = list(form_components.keys())
            self.logger.debug(f"Komponen yang tersedia di form_components: {available_components}")
            
            # Pastikan komponen yang diperlukan ada di form_components
            required_form_components = [
                'val_frequency_slider', 'iou_thres_slider', 'conf_thres_slider',
                'max_detections_slider', 'experiment_name_text', 'checkpoint_dir_text',
                'log_metrics_slider', 'visualize_batch_slider', 'gradient_clipping_slider',
                'layer_mode_dropdown', 'tensorboard_checkbox', 'multi_scale_checkbox',
                'img_size_min_slider', 'img_size_max_slider', 'save_button', 'reset_button'
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
                layout_components = create_strategy_layout(form_components)
                self.logger.debug("Layout components berhasil dibuat")
            except Exception as e:
                self.logger.error(f"Gagal membuat layout components: {str(e)}\n{traceback.format_exc()}")
                raise ValueError(f"Gagal membuat layout components: {str(e)}") from e
            
            # Debug: Tampilkan semua kunci yang tersedia di layout_components
            available_layout = list(layout_components.keys())
            self.logger.debug(f"Komponen yang tersedia di layout_components: {available_layout}")
            
            # Pastikan komponen yang diperlukan ada di layout_components
            required_layout_components = ['main_container', 'save_button', 'reset_button', 'summary_card']
            missing_layout = [comp for comp in required_layout_components if comp not in layout_components]
            if missing_layout:
                error_msg = (
                    f"Komponen layout yang hilang: {missing_layout}\n"
                    f"Komponen yang tersedia: {available_layout}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Pastikan main_container adalah widget yang valid
            if not isinstance(layout_components['main_container'], widgets.Widget):
                error_msg = f"main_container harus berupa instance widgets.Widget, tapi mendapat: {type(layout_components['main_container'])}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Update summary card dengan config terbaru
            try:
                update_summary_card(
                    layout_components,
                    config,
                    form_components
                )
                self.logger.debug("Summary card berhasil diupdate")
            except Exception as e:
                self.logger.error(f"Gagal mengupdate summary card: {str(e)}\n{traceback.format_exc()}")
                # Lanjutkan meskipun update summary card gagal
            
            # Return komponen yang diperlukan
            result = {
                'form': layout_components['main_container'],
                'save_button': layout_components['save_button'],
                'reset_button': layout_components['reset_button'],
                'summary_card': layout_components['summary_card'],
                'container': layout_components['main_container']
            }
            
            self.logger.debug("UI components berhasil dibuat")
            return result
                
        except Exception as e:
            self.logger.error(f"Error di _create_config_ui: {str(e)}\n{traceback.format_exc()}")
            return self.handle_ui_exception(e, context="UI strategy")
    
    def _setup_summary_update_callback(self, ui_components: Dict[str, Any]) -> None:
        """Setup callback untuk update summary card otomatis"""
        def update_summary_on_change(*args):
            """Callback untuk update summary saat ada perubahan config"""
            try:
                config_handler = ui_components.get('config_handler')
                if config_handler:
                    current_config = config_handler.extract_config(ui_components)
                    update_summary_card(ui_components, current_config)
            except Exception:
                pass  # Silent fail untuk callback
        
        # Register callback ke widget-widget penting
        key_widgets = [
            'epochs_slider', 'batch_size_slider', 'lr_slider', 'scheduler_dropdown',
            'mixed_precision_checkbox', 'tensorboard_checkbox', 'layer_mode_dropdown'
        ]
        
        [ui_components.get(widget_key) and hasattr(ui_components[widget_key], 'observe') and 
         ui_components[widget_key].observe(update_summary_on_change, names='value') 
         for widget_key in key_widgets if widget_key in ui_components]
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup custom handlers untuk strategy-specific functionality"""
        # Override parent method untuk menambahkan functionality khusus strategy
        pass


def initialize_strategy_config(env=None, config=None, parent_callbacks=None, **kwargs):
    """
    Factory function untuk strategy config cell dengan cascading inheritance
    
    Args:
        env: Environment manager instance
        config: Override config values (akan di-merge dengan cascading config)
        parent_callbacks: Callbacks untuk parent modules
        **kwargs: Additional arguments
        
    Returns:
        UI components dengan config yang sudah di-cascade
    """
    # Jika config tidak disediakan atau minimal, biarkan initializer load cascading config
    if not config:
        config = None  # Let initializer handle cascading loading
    
    return create_config_cell(
        StrategyInitializer, 
        'strategy', 
        'training_config', 
        env=env, 
        config=config, 
        config_handler_class=StrategyConfigHandler,
        parent_callbacks=parent_callbacks,
        **kwargs
    )


# Convenience function untuk direct initialization tanpa parent
def create_strategy_ui(config=None, **kwargs):
    """Create strategy UI secara langsung tanpa parent dependency"""
    return initialize_strategy_config(config=config, **kwargs)
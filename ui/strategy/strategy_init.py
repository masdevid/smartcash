"""
File: smartcash/ui/strategy/strategy_init.py
Deskripsi: Main initializer untuk strategy config cell dengan cascading inheritance support
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.strategy.handlers.config_handler import StrategyConfigHandler
from smartcash.ui.strategy.components.ui_form import create_strategy_form
from smartcash.ui.strategy.components.ui_layout import create_strategy_layout, update_summary_card
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.common.config.manager import get_config_manager


class StrategyInitializer(ConfigCellInitializer):
    """Strategy config cell initializer dengan cascading inheritance support"""
    
    def __init__(self, module_name='strategy', config_filename='training_config', 
                 config_handler_class=None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            config_handler_class = StrategyConfigHandler
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
        self.config_manager = get_config_manager()
    
    def _load_cascading_config(self) -> Dict[str, Any]:
        """Load config dengan cascading inheritance sesuai urutan yang benar"""
        config_manager = self.config_manager
        
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
                config = config_manager.get_config(config_name)
                if config:
                    # Remove _base_ untuk mencegah recursive inheritance
                    config.pop('_base_', None)
                    # Deep merge configs
                    merged_config = self._deep_merge_configs(merged_config, config)
                    self.logger.debug(f"🔗 Merged {config_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading {config_name}: {str(e)}")
        
        # Fallback ke defaults jika tidak ada config
        if not merged_config:
            merged_config = self.config_handler.get_default_config()
            self.logger.info("🔄 Using default config")
        
        return merged_config
    
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
            # Load cascading config jika config tidak disediakan atau minimal
            if not config:
                final_config = self._load_cascading_config()
                self.logger.info("🔗 Loaded cascading inheritance config")
            else:
                # Merge provided config dengan cascading config
                base_config = self._load_cascading_config()
                final_config = self._deep_merge_configs(base_config, config)
                self.logger.info("🔗 Merged provided config dengan cascading inheritance")
            
            # Call parent initialize dengan merged config
            return super().initialize(env=env, config=final_config, **kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ Error in strategy initialize: {str(e)}")
            # Fallback ke parent initialize dengan default config
            return super().initialize(env=env, config=self.config_handler.get_default_config(), **kwargs)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        try:
            # Load cascading config untuk memastikan defaults yang benar
            if not config or len(config) < 5:
                config = self._load_cascading_config()
            
            # Create form components dengan merged config
            form_components = create_strategy_form(config)
            
            # Create layout
            ui_components = create_strategy_layout(form_components)
            
            # Update summary card dengan config yang sudah merged
            update_summary_card(ui_components, config)
            
            # Setup callback untuk update summary saat config berubah
            self._setup_summary_update_callback(ui_components)
            
            return ui_components
        except Exception as e:
            self.logger.error(f"❌ Error creating training strategy UI: {str(e)}")
            # Fallback ke default config
            default_config = self.config_handler.get_default_config()
            form_components = create_strategy_form(default_config)
            ui_components = create_strategy_layout(form_components)
            update_summary_card(ui_components, default_config)
            return ui_components
    
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
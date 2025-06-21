
# File: smartcash/ui/hyperparameters/hyperparameters_init.py  
# Deskripsi: Fixed initializer untuk hyperparameters config dengan error handling

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.utils.fallback_utils import try_operation_safe, create_fallback_ui, FallbackConfig
from smartcash.ui.hyperparameters.handlers.defaults import get_default_hyperparameters_config, validate_hyperparameters_config
from smartcash.ui.hyperparameters.components.ui_form import HyperparametersForm
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class HyperparametersConfigInitializer(ConfigCellInitializer):
    """Fixed initializer untuk hyperparameters config 🔧"""
    
    def __init__(self, module_name='hyperparameters', config_filename='hyperparameters_config', 
                 config_handler_class=None, parent_module: Optional[str] = None):
        
        # Import handler inside __init__ to avoid circular imports
        if config_handler_class is None:
            try:
                from smartcash.ui.hyperparameters.handlers.config_handler import HyperparametersConfigHandler
                config_handler_class = HyperparametersConfigHandler
            except ImportError as e:
                logger.error(f"❌ Failed to import HyperparametersConfigHandler: {e}")
                config_handler_class = None
                
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters config 🖥️"""
        return try_operation_safe(
            operation=lambda: self._create_ui_components(config, env, **kwargs),
            fallback_value={
                'main_widget': create_fallback_ui(
                    error_message="Failed to create hyperparameters UI",
                    module_name=self.module_name,
                    config=FallbackConfig(
                        title="⚠️ Hyperparameters Error",
                        module_name=self.module_name
                    )
                )['main_widget'],
                'config': get_default_hyperparameters_config()
            },
            logger=logger,
            operation_name="creating hyperparameters UI"
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Internal logic untuk membuat UI components 🏗️"""
        logger.info("🚀 Creating hyperparameters UI components...")
        
        # Validasi config terlebih dahulu
        is_valid, error_msg = validate_hyperparameters_config(config)
        if not is_valid:
            logger.warning(f"⚠️ Invalid config, using defaults: {error_msg}")
            config = get_default_hyperparameters_config()
        
        # Buat form component
        form = HyperparametersForm(
            config=config.copy(),
            on_change=lambda: self._on_config_change(config)
        )
        
        # Buat action buttons
        action_buttons = self._create_action_buttons()
        
        # Buat main container
        main_widget = widgets.VBox([
            widgets.HTML("<h3>⚙️ Hyperparameters Configuration</h3>"),
            widgets.HTML("<p>Konfigurasi parameter training untuk model SmartCash</p>"),
            form.get_widget(),
            action_buttons
        ], layout=widgets.Layout(gap='15px', padding='10px'))
        
        return {
            'main_widget': main_widget,
            'form': form,
            'config': config,
            'action_buttons': action_buttons
        }
    
    def _create_action_buttons(self) -> widgets.HBox:
        """Buat action buttons untuk save/reset ⚡"""
        save_btn = widgets.Button(
            description='💾 Save Config',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        
        reset_btn = widgets.Button(
            description='🔄 Reset Default',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        save_btn.on_click(lambda x: self._handle_save())
        reset_btn.on_click(lambda x: self._handle_reset())
        
        return widgets.HBox([save_btn, reset_btn], layout=widgets.Layout(gap='10px'))
    
    def _on_config_change(self, config: Dict[str, Any]) -> None:
        """Handle perubahan config ⚡"""
        logger.info("🔄 Hyperparameters config changed")
        # Additional change handling can be added here
    
    def _handle_save(self) -> None:
        """Handle save button click 💾"""
        logger.info("💾 Saving hyperparameters config...")
        # Save logic will be handled by parent class
    
    def _handle_reset(self) -> None:
        """Handle reset button click 🔄"""
        logger.info("🔄 Resetting hyperparameters to defaults...")
        # Reset logic will be handled by parent class


def initialize_hyperparameters_config():
    """Initialize hyperparameters config cell - main entry point 🚀"""
    return try_operation_safe(
        operation=lambda: _initialize_hyperparameters_config(),
        fallback_value=create_fallback_ui(
            error_message="Failed to initialize hyperparameters config",
            module_name='hyperparameters',
            config=FallbackConfig(
                title="⚠️ Error in hyperparameters",
                module_name='hyperparameters'
            )
        ),
        logger=logger,
        operation_name="initializing hyperparameters configuration"
    )


def _initialize_hyperparameters_config():
    """Internal logic untuk initialize hyperparameters config 🔧"""
    logger.info("🚀 Initializing hyperparameters configuration...")
    
    # Buat initializer
    initializer = HyperparametersConfigInitializer()
    
    # Initialize config cell
    result = initializer.initialize()
    
    if result and result.get('success'):
        logger.info("✅ Hyperparameters configuration initialized successfully")
        return result
    else:
        raise Exception("Failed to initialize hyperparameters configuration")


def create_hyperparameters_config_cell():
    """Create hyperparameters config cell untuk direct usage 📋"""
    return try_operation_safe(
        operation=lambda: create_config_cell(
            module_name='hyperparameters',
            title='⚙️ Hyperparameters Configuration',
            description='Configure training hyperparameters untuk SmartCash model',
            initializer_class=HyperparametersConfigInitializer
        ),
        fallback_value=create_fallback_ui(
            error_message="Failed to create hyperparameters config cell",
            module_name='hyperparameters',
            config=FallbackConfig(
                title="⚠️ Error in hyperparameters",
                module_name='hyperparameters'
            )
        ),
        logger=logger,
        operation_name="creating hyperparameters config cell"
    )
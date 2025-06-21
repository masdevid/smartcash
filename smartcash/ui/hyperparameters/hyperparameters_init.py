# File: smartcash/ui/hyperparameters/hyperparameters_init.py
# Deskripsi: Main initializer untuk hyperparameters config cell - menggunakan fallback_utils

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.utils.fallback_utils import try_operation_safe, create_fallback_ui, FallbackConfig
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class HyperparametersConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk hyperparameters dengan inheritance dari config_cell_initializer"""
    
    def __init__(self, module_name='hyperparameters', config_filename='hyperparameters_config', 
                 config_handler_class=None, parent_module: Optional[str] = None):
        # Import handler di dalam __init__ untuk menghindari circular import
        if config_handler_class is None:
            from smartcash.ui.hyperparameters.handlers.config_handler import HyperparametersConfigHandler
            config_handler_class = HyperparametersConfigHandler
            
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk hyperparameters config dengan form dan layout"""
        return try_operation_safe(
            operation=lambda: self._create_ui_components(config, env, **kwargs),
            fallback_value=create_fallback_ui(
                error_message="Failed to create hyperparameters UI",
                module_name=self.module_name,
                config=FallbackConfig(
                    title=f"⚠️ Error in {self.module_name}",
                    module_name=self.module_name
                )
            ),
            logger=logger,
            operation_name="creating hyperparameters UI components"
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Internal logic untuk membuat UI components"""
        # Import components di dalam method untuk menghindari circular import
        from smartcash.ui.hyperparameters.components.ui_form import create_hyperparameters_form
        from smartcash.ui.hyperparameters.components.ui_layout import create_hyperparameters_layout
        
        # Pastikan config memiliki struktur yang benar
        if not isinstance(config, dict):
            config = {}
            
        # Inisialisasi section yang diperlukan
        required_sections = ['training', 'optimizer', 'scheduler', 'loss', 'early_stopping', 'checkpoint']
        for section in required_sections:
            if section not in config:
                config[section] = {}
        
        # Buat form widgets
        form_widgets = create_hyperparameters_form(config)
        
        # Buat layout UI
        ui_layout = create_hyperparameters_layout(form_widgets)
        
        # Return widgets dan layout
        return {
            'widgets': form_widgets,
            'layout': ui_layout,
            'config': config
        }


def initialize_hyperparameters_config():
    """Initialize hyperparameters configuration cell 🎯"""
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
    """Internal logic untuk initialize hyperparameters config"""
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
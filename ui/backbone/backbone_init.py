"""
File: smartcash/ui/backbone/backbone_init.py
Deskripsi: Initializer untuk backbone configuration menggunakan backbone_config.yaml dengan pewarisan config_cell_initializer
"""

import traceback
import sys
from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.common.logger import get_logger

logger = get_logger(__name__)
MODULE_NAME = 'backbone'
MODULE_CONFIG = MODULE_NAME+"_config"
class BackboneInitializer(ConfigCellInitializer):
    """Config cell initializer untuk backbone configuration menggunakan backbone_config.yaml"""
    
    def __init__(self, module_name=MODULE_NAME, config_filename=MODULE_CONFIG, config_handler_class=None, parent_module=None):
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk backbone configuration dengan reusable components"""
        try:
            from .components.ui_form import create_backbone_form
            from .components.ui_layout import create_backbone_layout
            from ..utils.fallback_utils import create_fallback_ui
            
            # Create form components menggunakan backbone_config.yaml
            form_components = create_backbone_form(config)
            
            # Create layout dengan form components
            return create_backbone_layout(form_components)
            
        except Exception as e:
            return self.handle_ui_exception(e, context="UI backbone configuration")
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> None:
        """Setup custom handlers untuk backbone selection changes"""
        from .components.selection_change import setup_backbone_selection_handlers
        setup_backbone_selection_handlers(ui_components, config)

def initialize_backbone_config(env=None, config=None, parent_callbacks=None, **kwargs) -> Any:
    """
    Factory function untuk backbone config cell dengan parent integration.
    
    Args:
        env: Environment manager instance
        config: Override config values
        parent_callbacks: Dict callbacks untuk parent modules seperti training
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI jika gagal
    """
    from .handlers.config_handler import BackboneConfigHandler
    
    return create_config_cell(
        BackboneInitializer, 
        MODULE_NAME, 
        MODULE_CONFIG,
        env=env, 
        config=config, 
        config_handler_class=BackboneConfigHandler,
        parent_callbacks=parent_callbacks,
        **kwargs
    )
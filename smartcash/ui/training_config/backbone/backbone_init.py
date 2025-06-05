"""
File: smartcash/ui/training_config/backbone/backbone_init.py
Deskripsi: Config cell initializer untuk backbone model configuration dengan DRY pattern
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.handlers.config_handlers import ConfigHandler

class BackboneConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk backbone model configuration"""
    
    def __init__(self, module_name='backbone', config_filename='model', config_handler_class=None, 
                 parent_module: Optional[str] = 'training'):
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk backbone configuration"""
        from smartcash.ui.training_config.backbone.components.ui_form import create_backbone_form
        from smartcash.ui.training_config.backbone.components.ui_layout import create_backbone_layout
        
        # Create form components
        form_components = create_backbone_form(config)
        
        # Create layout dengan form components
        return create_backbone_layout(form_components)
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> None:
        """Setup custom handlers untuk backbone UI"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import setup_backbone_handlers
        setup_backbone_handlers(ui_components, config)



class BackboneConfigHandler(ConfigHandler):
    """Config handler untuk backbone model configuration"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        from smartcash.ui.training_config.backbone.handlers.config_extractor import extract_backbone_config
        return extract_backbone_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        from smartcash.ui.training_config.backbone.handlers.config_updater import update_backbone_ui
        update_backbone_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default backbone configuration"""
        from smartcash.ui.training_config.backbone.handlers.defaults import get_default_backbone_config
        return get_default_backbone_config()


def initialize_backbone_config(env=None, config=None, parent_callbacks=None, **kwargs) -> Any:
    """
    Factory function untuk create backbone config cell.
    
    Args:
        env: Environment manager instance
        config: Override config values
        parent_callbacks: Callbacks untuk parent modules (training, evaluation)
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI
    """
    return create_config_cell(
        BackboneConfigInitializer, 
        'backbone', 
        'model', 
        env=env, 
        config=config, 
        config_handler_class=BackboneConfigHandler,
        parent_module='training',
        parent_callbacks=parent_callbacks,
        **kwargs
    )
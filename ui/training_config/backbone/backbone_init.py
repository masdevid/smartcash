"""
File: smartcash/ui/training_config/backbone/backbone_init.py
Deskripsi: Config cell initializer untuk backbone model configuration dengan DRY pattern
"""

from typing import Dict, Any
from smartcash.ui.utils.config_cell_initializer import ConfigCellInitializer, create_config_cell

class BackboneConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk backbone model configuration"""
    
    def __init__(self, module_name='backbone', config_filename='model'):
        super().__init__(module_name, config_filename)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk backbone configuration"""
        from smartcash.ui.training_config.backbone.components.backbone_form import create_backbone_form
        from smartcash.ui.training_config.backbone.components.backbone_layout import create_backbone_layout
        
        # Create form components
        form_components = create_backbone_form(config)
        
        # Create layout dengan form components
        return create_backbone_layout(form_components)
    
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        from smartcash.ui.training_config.backbone.handlers.ui_extractor import extract_backbone_config
        return extract_backbone_config(ui_components)
    
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        from smartcash.ui.training_config.backbone.handlers.ui_updater import update_backbone_ui
        update_backbone_ui(ui_components, config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default backbone configuration"""
        from smartcash.ui.training_config.backbone.handlers.defaults import get_default_backbone_config
        return get_default_backbone_config()
    
    def _setup_custom_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> None:
        """Setup custom handlers untuk backbone UI"""
        from smartcash.ui.training_config.backbone.handlers.form_handlers import setup_backbone_handlers
        setup_backbone_handlers(ui_components, config)

def initialize_backbone_config(env=None, config=None, **kwargs) -> Any:
    """
    Factory function untuk create backbone config cell.
    
    Args:
        env: Environment manager instance
        config: Override config values
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI
    """
    return create_config_cell(BackboneConfigInitializer, 'backbone', 'model', env, config, **kwargs)
"""
File: smartcash/ui/backbone/backbone_init.py
Deskripsi: Initializer untuk backbone configuration menggunakan backbone_config.yaml dengan pewarisan config_cell_initializer
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell

class BackboneInitializer(ConfigCellInitializer):
    """Config cell initializer untuk backbone configuration menggunakan backbone_config.yaml"""
    
    def __init__(self):
        super().__init__('backbone', 'backbone_config')
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk backbone configuration dengan reusable components"""
        from .components.ui_form import create_backbone_form
        from .components.ui_layout import create_backbone_layout
        
        # Create form components menggunakan backbone_config.yaml
        form_components = create_backbone_form(config)
        
        # Create layout dengan form components
        return create_backbone_layout(form_components)
    
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
        'backbone', 
        'backbone_config', 
        env=env, 
        config=config, 
        config_handler_class=BackboneConfigHandler,
        parent_callbacks=parent_callbacks,
        **kwargs
    )
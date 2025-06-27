"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Initializer untuk dependency management module
"""

from typing import Dict, Any, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler

class DependencyInitializer(CommonInitializer):
    """Initializer untuk dependency management"""
    
    def __init__(self, module_name: str = 'dependency', 
                 config_handler_class: Type[ConfigHandler] = DependencyConfigHandler,
                 **kwargs):
        super().__init__(
            module_name=module_name,
            config_handler_class=config_handler_class,
            **kwargs
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create dependency management UI components"""
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        
        ui_components = create_dependency_main_ui(config)
        
        if not ui_components or 'ui' not in ui_components:
            raise ValueError("UI components tidak valid atau kosong")
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default dependency configuration"""
        from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
        return get_default_dependency_config()
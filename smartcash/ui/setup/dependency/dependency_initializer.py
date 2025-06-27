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
        
    def initialize_ui(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize the dependency management UI
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Dictionary containing UI components
        """
        # Load default config if none provided
        if config is None:
            config = self._get_default_config()
            
        # Create UI components
        ui_components = self._create_ui_components(config)
        
        # Setup event handlers
        from smartcash.ui.setup.dependency.handlers.event_handlers import setup_all_handlers
        handlers = setup_all_handlers(ui_components, config, self.config_handler)
        ui_components['_handlers'] = handlers
        
        # Initialize logger bridge if not present
        if 'logger_bridge' not in ui_components:
            from smartcash.ui.utils.logger_bridge import LoggerBridge
            ui_components['logger_bridge'] = LoggerBridge(
                log_output=ui_components.get('log_output'),
                summary_output=ui_components.get('status_panel'),
                module_name='dependency'
            )
            
        # Log successful initialization
        ui_components['logger_bridge'].info("✅ Dependency management UI initialized")
        
        return ui_components
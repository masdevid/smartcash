"""
File: smartcash/ui/model/backbone/backbone_initializer.py
Description: Backbone Model Configuration Initializer following core UI structure
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.initializers.display_initializer import create_ui_display_function
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from .configs.backbone_defaults import get_default_backbone_config
from .handlers.backbone_ui_handler import BackboneUIHandler


class BackboneInitializer(ModuleInitializer):
    """Initializer for backbone module following core UI structure."""
    
    def __init__(self, module_name: str = 'backbone', parent_module: Optional[str] = 'model'):
        """Initialize backbone module initializer.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
        """
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            handler_class=BackboneUIHandler,
            auto_setup_handlers=True
        )
        self._ui_components = None
        self._backbone_factory = None
        self.logger.info(f"🛠️ BackboneInitializer created for module: {module_name}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default backbone configuration."""
        return get_default_backbone_config()
    
    def pre_initialize_checks(self) -> None:
        """Pre-initialization validation checks."""
        # Check if required imports are available
        try:
            from .components.ui_components import create_backbone_ui_components
            from .handlers.backbone_ui_handler import BackboneUIHandler
        except ImportError as e:
            raise RuntimeError(f"Missing required components: {e}")
    
    def post_initialize_cleanup(self) -> None:
        """Post-initialization cleanup and validation."""
        # Validate that essential UI components were created
        if not self._ui_components:
            raise RuntimeError("No UI components were created")
        
        required_components = ['main_container', 'ui']
        missing = [comp for comp in required_components if comp not in self._ui_components]
        if missing:
            self.logger.warning(f"⚠️ Missing components: {missing}")
    
    def setup_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup module handler with UI components.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.logger.info("🔧 Setting up backbone module handlers...")
        
        # Validate input
        if not ui_components:
            error_msg = "No UI components provided for handler setup"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Create module handler if it doesn't exist
            if not hasattr(self, '_module_handler') or self._module_handler is None:
                self.logger.debug("Creating new module handler instance")
                self._module_handler = self.create_module_handler()
            
            # Setup handler with UI components
            self.logger.debug("Setting up module handler with UI components")
            self._module_handler.setup(ui_components)
            
            # Initialize handlers dictionary
            if not hasattr(self, '_handlers') or not isinstance(self._handlers, dict):
                self._handlers = {}
            
            # Register handlers
            self._handlers.update({
                'module': self._module_handler,
                'config': self._module_handler  # Alias for backward compatibility
            })
            
            self.logger.info(f"✅ Successfully set up {len(self._handlers)} handlers")
            
        except Exception as e:
            error_msg = f"Failed to set up handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def setup_backbone_factory(self) -> None:
        """Setup backbone factory for operations."""
        try:
            import sys
            sys.path.append('.')
            from model.utils.backbone_factory import BackboneFactory
            self._backbone_factory = BackboneFactory()
            self.logger.info("✅ Backbone factory initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize backbone factory: {e}")
            self._backbone_factory = None
    
    def _initialize_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """Implementation of initialization logic.
        
        Returns:
            Dict containing initialization results
        """
        # Extract config from args/kwargs
        config = None
        if args:
            config = args[0]
        elif 'config' in kwargs:
            config = kwargs['config']
        
        if config is None:
            config = self.get_default_config()
        
        try:
            self.logger.info("🚀 Starting backbone module initialization...")
            
            # Pre-initialization phase
            self.pre_initialize_checks()
            
            # UI components phase
            from .components.ui_components import create_backbone_ui_components
            ui_components = create_backbone_ui_components(config)
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            self._ui_components = ui_components
            
            # Handlers setup phase (including operation container setup)
            self.setup_handlers(ui_components)
            
            # Backbone factory setup phase
            self.setup_backbone_factory()
            
            # Post-initialization phase
            self.post_initialize_cleanup()
            
            self.logger.info("✅ Backbone module initialized successfully")
            
            return {
                'success': True,
                'ui_components': self._ui_components,
                'module_handler': self._module_handler,
                'config_handler': getattr(self, 'config_handler', None),
                'backbone_factory': self._backbone_factory,
                'config': config
            }
            
        except Exception as e:
            from smartcash.ui.core.errors import get_error_handler
            error_handler = get_error_handler()
            error_handler.handle_exception(e, 'initialization', fail_fast=False)
            self.logger.error(f"❌ Initialization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ui_components': {},
                'module_handler': None,
                'config_handler': None,
                'backbone_factory': None
            }


def _backbone_initialize_legacy(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Legacy initialization function for backbone module."""
    initializer = BackboneInitializer()
    return initializer._initialize_impl(config, **kwargs)


# Create the display function following core UI structure pattern
initialize_backbone_ui = create_ui_display_function(
    module_name='backbone',
    parent_module='model',
    initializer_class=BackboneInitializer,
    legacy_function=_backbone_initialize_legacy
)


# Global instance for backward compatibility
_backbone_initializer: Optional[BackboneInitializer] = None


def get_backbone_initializer() -> BackboneInitializer:
    """Get or create backbone initializer instance.
    
    Returns:
        BackboneInitializer instance
    """
    global _backbone_initializer
    
    if _backbone_initializer is None:
        _backbone_initializer = BackboneInitializer()
    
    return _backbone_initializer


def init_backbone_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Initialize backbone UI module.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of UI components
    """
    return initialize_backbone_ui(config, **kwargs)
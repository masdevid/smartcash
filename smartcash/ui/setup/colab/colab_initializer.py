"""
File: smartcash/ui/setup/colab/colab_initializer.py
Description: Colab Environment Configuration Initializer following dependency pattern
"""

import contextlib
from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from .configs.colab_defaults import get_default_colab_config
from .handlers.colab_ui_handler import ColabUIHandler
from .components import create_colab_ui_components



class ColabInitializer(ModuleInitializer):
    """Initializer for colab module following dependency pattern."""
    
    def __init__(self, module_name: str = 'colab', parent_module: Optional[str] = 'setup'):
        """Initialize colab module initializer.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
        """
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            handler_class=ColabUIHandler,
            auto_setup_handlers=True
        )
        self._ui_components = None
        self._operation_handlers = {}
        self._environment_manager = None
        self.logger.info(f"🛠️ ColabInitializer created for module: {module_name}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default colab configuration."""
        return get_default_colab_config()
    
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
            self.logger.info("🚀 Starting colab module initialization...")
            
            # Pre-initialization phase
            self.pre_initialize_checks()
            
            # UI components phase
            ui_components = create_colab_ui_components(config)
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            self._ui_components = ui_components
            
            # Handlers setup phase
            self.setup_handlers(ui_components)
            
            # Operation handlers setup phase
            self.setup_operation_handlers()
            
            # Environment manager setup phase
            self.setup_environment_manager()
            
            # Post-initialization phase
            self.post_initialize_cleanup()
            
            self.logger.info("✅ Colab module initialized successfully")
            
            return {
                'success': True,
                'ui_components': self._ui_components,
                'module_handler': self._module_handler,
                'config_handler': getattr(self, 'config_handler', None),
                'operation_handlers': self._operation_handlers,
                'config': config
            }
            
        except Exception as e:
            from smartcash.ui.core.shared.error_handler import get_error_handler
            error_handler = get_error_handler('colab')
            error_handler.handle_exception(e, 'initialization', fail_fast=False)
            self.logger.error(f"❌ Initialization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ui_components': {},
                'module_handler': None,
                'config_handler': None,
                'operation_handlers': {}
            }
    
    def pre_initialize_checks(self) -> None:
        """Pre-initialization validation checks."""
        # Check if required imports are available
        try:
            from .components import create_colab_ui_components
            from .handlers.colab_ui_handler import ColabUIHandler
            from .configs.colab_config_handler import ColabConfigHandler
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
        self.logger.info("🔧 Setting up colab module handlers...")
        
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
    
    def setup_operation_handlers(self) -> None:
        """Setup colab-specific operation handlers."""
        try:
            from .operations.factory import OperationHandlerFactory
            
            # Create operation handlers for colab-specific operations
            self._operation_handlers = {
                'environment_detection': OperationHandlerFactory.create_handler('environment_detection', self._ui_components, self.get_default_config()),
                'drive_mount': OperationHandlerFactory.create_handler('drive_mount', self._ui_components, self.get_default_config()),
                'gpu_setup': OperationHandlerFactory.create_handler('gpu_setup', self._ui_components, self.get_default_config()),
                'folder_setup': OperationHandlerFactory.create_handler('folder_setup', self._ui_components, self.get_default_config()),
                'config_sync': OperationHandlerFactory.create_handler('config_sync', self._ui_components, self.get_default_config()),
                'verify': OperationHandlerFactory.create_handler('verify', self._ui_components, self.get_default_config())
            }
            
            self.logger.info("✅ Operation handlers setup complete")
            
        except ImportError:
            self.logger.warning("⚠️ Operation handlers not available, creating minimal setup")
            self._operation_handlers = {}
        except Exception as e:
            from smartcash.ui.core.shared.error_handler import get_error_handler
            error_handler = get_error_handler('colab')
            error_handler.handle_exception(e, 'setting up operation handlers', fail_fast=False)
    
    def setup_environment_manager(self) -> None:
        """Setup environment manager for operations."""
        try:
            from smartcash.common.environment import get_environment_manager
            self._environment_manager = get_environment_manager()
            self.logger.info("✅ Environment manager initialized")
        except ImportError:
            self.logger.warning("⚠️ Environment manager not available")
            self._environment_manager = None
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize environment manager: {e}")
            self._environment_manager = None


# Global instance for backward compatibility
_colab_initializer: Optional[ColabInitializer] = None


def initialize_colab_ui_internal(config: Optional[Dict[str, Any]] = None) -> Any:
    """Internal colab UI initialization that returns components.
    
    This function ensures only one instance of the colab UI is created.
    
    Args:
        config: Optional configuration dictionary for initialization
        
    Returns:
        The main UI container widget or dict of components
    """
    global _colab_initializer
    
    # If we already have an initialized instance, return its UI
    if (_colab_initializer is not None and 
        hasattr(_colab_initializer, '_ui_components') and 
        _colab_initializer._ui_components is not None):
        return _colab_initializer._ui_components.get('ui')
    elif _colab_initializer is not None:
        # If _ui_components is None, force reinitialization
        _colab_initializer = None
    
    # Otherwise, create a new instance using ModuleInitializer
    from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
    
    # Use the centralized initialization
    result = ModuleInitializer.initialize_module_ui(
        module_name='colab',
        parent_module='setup',
        config=config,
        initializer_class=ColabInitializer
    )
    
    # Store the initializer instance for future use
    _colab_initializer = ModuleInitializer.get_module_instance('colab', 'setup')
    
    return result


# Import the display function creator
from smartcash.ui.core.initializers.display_initializer import create_ui_display_function

# Create the initialize function using the consistent pattern with legacy fallback
initialize_colab_ui = create_ui_display_function(
    module_name='colab',
    parent_module='setup',
    legacy_function=initialize_colab_ui_internal
)


def get_colab_initializer() -> ColabInitializer:
    """Get or create colab initializer instance.
    
    Returns:
        ColabInitializer instance
    """
    global _colab_initializer
    
    if _colab_initializer is None:
        _colab_initializer = ColabInitializer()
    
    return _colab_initializer


# Legacy support
ColabEnvInitializer = ColabInitializer

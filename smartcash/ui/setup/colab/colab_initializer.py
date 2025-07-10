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
            auto_setup_handlers=True,
            enable_shared_config=False  # Disable shared config for colab (no persistence)
        )
        self._ui_components = None
        self._operation_handlers = {}
        self._environment_manager = None
        self.logger.info(f"🛠️ ColabInitializer created for module: {module_name}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default colab configuration."""
        return get_default_colab_config()
    
    def load_config(self, name: str = None) -> bool:
        """Override to disable persistent config loading for colab module.
        
        Colab module does not use persistent configuration as per module structure docs.
        Always returns True to indicate successful "loading" of default config.
        
        Args:
            name: Config name (ignored)
            
        Returns:
            True (always successful with default config)
        """
        # Simply use default config - no file loading
        self.config = self.get_default_config()
        self.logger.debug("📋 Using default colab config (no persistence)")
        return True
    
    def save_config(self) -> bool:
        """Override to disable config saving for colab module.
        
        Colab module does not use persistent configuration.
        
        Returns:
            True (no-op operation always succeeds)
        """
        self.logger.debug("📋 Colab config save disabled (no persistence)")
        return True
    
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
            from smartcash.ui.core.errors.handlers import get_error_handler
            error_handler = get_error_handler()
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
            config = self.get_default_config()
            self._operation_handlers = {
                'environment_detection': OperationHandlerFactory.create_handler(
                    'environment_detection', self._ui_components, config
                ),
                'drive_mount': OperationHandlerFactory.create_handler(
                    'drive_mount', self._ui_components, config
                ),
                'gpu_setup': OperationHandlerFactory.create_handler(
                    'gpu_setup', self._ui_components, config
                ),
                'folder_setup': OperationHandlerFactory.create_handler(
                    'folder_setup', self._ui_components, config
                ),
                'config_sync': OperationHandlerFactory.create_handler(
                    'config_sync', self._ui_components, config
                ),
                'verify': OperationHandlerFactory.create_handler(
                    'verify', self._ui_components, config
                )
            }
            
            self.logger.info("✅ Operation handlers setup complete")
            
        except ImportError:
            self.logger.warning("⚠️ Operation handlers not available, creating minimal setup")
            self._operation_handlers = {}
        except Exception as e:
            from smartcash.ui.core.errors.handlers import get_error_handler
            error_handler = get_error_handler()
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

# Add DisplayInitializer import
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer

class ColabDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for colab module"""
    
    def __init__(self):
        super().__init__(module_name="colab", parent_module="setup")
        self._colab_initializer = ColabInitializer()
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation using existing ColabInitializer"""
        return self._colab_initializer._initialize_impl(**kwargs)

# Global display initializer instance
_colab_display_initializer = ColabDisplayInitializer()

def initialize_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Initialize and display colab UI using DisplayInitializer

    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    
    Note:
        This function displays the UI directly and returns None.
        Use get_colab_components() if you need access to the components dictionary.
    """
    _colab_display_initializer.initialize_and_display(config=config, **kwargs)

def get_colab_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Get colab components dictionary without displaying UI

    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components
    """
    global _colab_initializer
    if _colab_initializer is None:
        _colab_initializer = ColabInitializer()
    return _colab_initializer.initialize(config=config, **kwargs)

def display_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Display colab UI (alias for initialize_colab_ui)

    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_colab_ui(config=config, **kwargs)


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

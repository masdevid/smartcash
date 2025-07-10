"""
File: smartcash/ui/setup/colab/colab_initializer.py
Description: Colab Environment Configuration Initializer following dependency pattern

Note: This module uses lazy loading to prevent automatic setup on import.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.logging import suppress_ui_initialization_logs, setup_ui_logging
from .configs.colab_defaults import get_default_colab_config

# Global instance (lazy initialized)
_colab_initializer = None

def get_colab_initializer() -> 'ColabInitializer':
    """Get or create colab initializer instance.
    
    Note: This is the only function that should be used to get the initializer.
    It ensures proper lazy initialization of all components.
    
    Returns:
        ColabInitializer instance
    """
    global _colab_initializer
    
    if _colab_initializer is None:
        # Import here to prevent circular imports
        from .handlers.colab_ui_handler import ColabUIHandler
        from .components import create_colab_ui
        
        # Set the global components for the module
        globals()['ColabUIHandler'] = ColabUIHandler
        globals()['create_colab_ui'] = create_colab_ui
        
        # Create the initializer
        _colab_initializer = ColabInitializer()
    
    return _colab_initializer

# Lazy imports to prevent circular imports and automatic setup
if TYPE_CHECKING:
    from .handlers.colab_ui_handler import ColabUIHandler
    from .components.colab_ui import create_colab_ui

# Legacy support
get_colab_env_initializer = get_colab_initializer


class ColabInitializer(ModuleInitializer):
    """Initializer for colab module following dependency pattern."""
    
    def __init__(self, module_name: str = 'colab', parent_module: Optional[str] = 'setup'):
        """Initialize colab module initializer.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
        """
        # Initialize with no persistence and disable auto-setup
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            handler_class=ColabUIHandler,
            config_handler_class=None,  # Use default in-memory config handler
            enable_shared_config=False,  # Disable shared config
            auto_setup_handlers=False  # Disable auto-setup to prevent premature execution
        )
        
        # Initialize instance variables
        self._ui_components = None
        self._operation_handlers = {}
        self._environment_manager = None
        
        # Set default config directly
        self._config = self.get_default_config()
        
        # Disable any config persistence
        self.config_handler = None
        
        self.logger.info(f"🛠️ ColabInitializer created for module: {module_name}")
        
        # Override config property to use our local _config
        self.config = self._config
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return getattr(self, '_config', {})
    
    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """Set configuration."""
        if not hasattr(self, '_config'):
            self._config = {}
        self._config.update(value)
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default colab configuration."""
        return get_default_colab_config()
    
    def load_config(self, name: str = None) -> bool:
        """Override to disable persistent config loading for colab module.
        
        Colab module does not use persistent configuration.
        Always returns True to indicate successful "loading" of default config.
        
        Args:
            name: Config name (ignored)
            
        Returns:
            True (always successful with default config)
        """
        # No-op - config is already set in __init__
        return True
    
    def save_config(self) -> bool:
        """Override to disable config saving for colab module.
        
        Colab module does not use persistent configuration.
        
        Returns:
            True (no-op operation always succeeds)
        """
        # No-op - config is never persisted
        return True
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create colab UI components
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components with required structure
        """
        try:
            self.logger.info("🔧 Creating colab UI components")
            
            # Create UI components
            ui_components = create_colab_ui(config=config, **kwargs)
            
            if not isinstance(ui_components, dict):
                raise ValueError("create_colab_ui() did not return a dictionary")
                
            # Ensure required components exist
            required_components = ['main_container', 'operation_container']
            for comp in required_components:
                if comp not in ui_components:
                    raise ValueError(f"Missing required UI component: {comp}")
            
            # Add module-specific metadata
            ui_components.update({
                'colab_initialized': True,
                'module_name': 'colab',
                'logger': self.logger,
                'config': config,
                'ui': ui_components.get('main_container')  # Ensure 'ui' key exists
            })
            
            self.logger.info(f"✅ UI components created successfully: {', '.join(ui_components.keys())}")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create UI components: {str(e)}")
            raise
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup module handlers and configure logging to route to UI components
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Updated UI components with handlers
        """
        try:
            self.logger.info("🔧 Setting up colab module handlers")
            
            # Validate UI components
            if not ui_components or not isinstance(ui_components, dict):
                raise ValueError("Invalid UI components dictionary")
                
            # Ensure operation container exists
            operation_container = ui_components.get('operation_container')
            if not operation_container:
                raise ValueError("Missing operation_container in UI components")
            
            # Setup global UI logging handler to route logs to UI log_output only
            if 'log_message' in ui_components:
                setup_ui_logging('colab', ui_components['log_message'])
            
            # Setup module handlers with UI components but don't execute setup
            self.setup_handlers(ui_components)
            
            # Initialize operation handlers without executing them
            self.setup_operation_handlers()
            
            # Initialize environment manager if available
            self.setup_environment_manager()
            
            # Ensure the module handler has the latest UI components
            if hasattr(self, '_module_handler') and self._module_handler:
                self._module_handler.ui_components = ui_components
                
            # Explicitly set setup_in_progress to False to ensure clean state
            if hasattr(self, '_module_handler') and hasattr(self._module_handler, 'setup_in_progress'):
                self._module_handler.setup_in_progress = False
            
            self.logger.info("✅ Module handlers setup complete")
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Failed to setup module handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Try to log to operation container if available
            if 'log_message' in ui_components:
                try:
                    ui_components['log_message'](error_msg, 'ERROR')
                except:
                    pass
            raise
    
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of initialization logic
        
        Returns:
            Dictionary containing initialization results with UI components
            
        Raises:
            RuntimeError: If initialization fails
        """
        config = None
        ui_components = None
        
        try:
            # Get or create config
            config = kwargs.get('config') or self.get_default_config()
            if 'config' in kwargs:
                kwargs.pop('config')
            
            self.logger.info("🚀 Starting colab module initialization...")
            self.pre_initialize_checks()
            
            # Create UI components
            ui_components = self._create_ui_components(config=config, **kwargs)
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Setup module handlers
            self._setup_module_handlers(ui_components=ui_components, config=config, **kwargs)
            
            # Post-initialization cleanup
            self.post_initialize_cleanup()
            
            # Get the main container and operation container
            main_container = ui_components.get('ui') or ui_components.get('main_container')
            operation_container = ui_components.get('operation_container')
            
            if not main_container:
                raise ValueError("No main container found in UI components")
            
            self.logger.info("✅ Colab module initialized successfully")
            
            # Return components in the format expected by DisplayInitializer
            result = {
                'success': True,
                'ui': main_container,
                'ui_components': ui_components,
                'operation_container': operation_container,
                'module_handler': getattr(self, '_module_handler', None),
                'config': config
            }
            
            # Log the result structure for debugging
            self.logger.debug(f"Initialization result keys: {', '.join(result.keys())}")
            return result
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize colab module: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to log to operation container if available
            if ui_components and 'operation_container' in ui_components:
                try:
                    ui_components['operation_container'].log_error(error_msg)
                except:
                    pass
                    
            # Return error information in the expected format
            return {
                'success': False,
                'error': error_msg,
                'ui_components': ui_components or {},
                'config': config or {}
            }
    
    def pre_initialize_checks(self) -> None:
        """Pre-initialization validation checks."""
        # Check if required imports are available
        try:
            from .components import create_colab_ui
            from .handlers.colab_ui_handler import ColabUIHandler
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
            
        Raises:
            RuntimeError: If handler setup fails
        """
        try:
            self.logger.info("🔧 Setting up colab module handlers...")
            
            # Validate input
            if not ui_components or not isinstance(ui_components, dict):
                error_msg = "Invalid UI components provided for handler setup"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Store UI components reference
            self._ui_components = ui_components
            self.logger.debug(f"📦 Stored UI components: {', '.join(ui_components.keys())}")
            
            # Create module handler if it doesn't exist
            if not hasattr(self, '_module_handler') or not self._module_handler:
                self.logger.info("🛠️ Creating new module handler instance")
                self._module_handler = self.create_module_handler()
            
            # Setup handler with UI components
            self.logger.debug("Setting up module handler with UI components")
            if hasattr(self._module_handler, 'setup'):
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
            
            # Setup operation handlers now that we have a module handler
            self.setup_operation_handlers()
            
        except Exception as e:
            error_msg = f"❌ Failed to set up handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to log to operation container if available
            if 'log_message' in ui_components:
                try:
                    ui_components['log_message'](error_msg, 'ERROR')
                except:
                    pass
                    
            raise RuntimeError(error_msg) from e
    
    def setup_operation_handlers(self) -> None:
        """Setup colab-specific operation handlers."""
        try:
            from .operations.factory import OperationHandlerFactory
            
            # Skip if UI components are not yet initialized
            if not hasattr(self, '_ui_components') or not self._ui_components:
                self.logger.warning("⚠️ UI components not yet initialized, skipping operation handlers setup")
                self._operation_handlers = {}
                return
                
            # Create operation handlers for colab-specific operations
            config = self.get_default_config()
            
            # Initialize operation handlers dictionary
            self._operation_handlers = {}
            
            # Define the operation types we want to create handlers for
            operation_types = [
                'environment_detection',
                'drive_mount',
                'gpu_setup',
                'folder_setup',
                'config_sync',
                'verify'
            ]
            
            # Create handlers for each operation type
            for op_type in operation_types:
                try:
                    self._operation_handlers[op_type] = OperationHandlerFactory.create_handler(
                        op_type, self._ui_components, config
                    )
                    self.logger.debug(f"✅ Created operation handler for: {op_type}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to create operation handler for {op_type}: {str(e)}")
            
            if not self._operation_handlers:
                self.logger.warning("⚠️ No operation handlers were created successfully")
            else:
                self.logger.info(f"✅ Successfully created {len(self._operation_handlers)} operation handlers")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Operation handlers not available: {str(e)}")
            self._operation_handlers = {}
        except Exception as e:
            error_msg = f"❌ Failed to setup operation handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to log to operation container if available
            if hasattr(self, '_ui_components') and self._ui_components and 'log_message' in self._ui_components:
                try:
                    self._ui_components['log_message'](error_msg, 'ERROR')
                except:
                    pass
                    
            self._operation_handlers = {}
    
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


class ColabDisplayInitializer:
    """DisplayInitializer wrapper for colab module"""
    
    def __init__(self):
        """Initialize the display initializer."""
        # Don't initialize the base class with parameters
        # as we're not actually inheriting from DisplayInitializer
        self._colab_initializer = get_colab_initializer()
        self._cached_components = None
    
    def _initialize_impl(self, **kwargs):
        """Implementation using existing ColabInitializer
        
        Returns:
            Dictionary containing UI components and initialization results
        """
        # Delegate to the colab initializer
        result = self._colab_initializer._initialize_impl(**kwargs)
        
        # Add the initializer to the result
        if isinstance(result, dict):
            result['initializer'] = self._colab_initializer
        
        return result
        
    def display(self, config=None, **kwargs):
        """Display the colab UI with global logging suppression.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional arguments
        """
        from IPython.display import display, clear_output
        
        # Clear any previous output to avoid caching issues
        clear_output(wait=True)
        
        # Use global logging suppression
        is_test_mode = kwargs.get('_test_mode', False)
        with suppress_ui_initialization_logs(test_mode=is_test_mode):
            try:
                result = self._colab_initializer.initialize(config=config, **kwargs)
                if result and 'ui_components' in result:
                    ui_components = result['ui_components']
                    # Cache the components
                    self._cached_components = ui_components
                    # Try different possible UI widget keys
                    ui_widget = ui_components.get('ui') or ui_components.get('main_container') or ui_components.get('container')
                    if ui_widget:
                        display(ui_widget)
                    else:
                        if not is_test_mode:
                            print("⚠️ No displayable UI widget found in components")
                else:
                    if not is_test_mode:
                        print("⚠️ No result or ui_components from initializer")
            except Exception as e:
                # Only show errors if not in test mode
                if not is_test_mode:
                    print(f"❌ Error displaying colab UI: {str(e)}")
                raise
    
    def get_components(self, config=None, **kwargs):
        """Get the UI components without displaying them.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        result = self._colab_initializer.initialize(config=config, **kwargs)
        return result.get('ui_components', {}) if result else {}


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
    _colab_display_initializer.display(config=config, **kwargs)


def get_colab_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Get colab components dictionary without displaying UI
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components
    """
    # Create a new display initializer to avoid widget caching
    display_initializer = ColabDisplayInitializer()
    return display_initializer.get_components(config=config, **kwargs)


def display_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Display colab UI (alias for initialize_colab_ui)
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_colab_ui(config=config, **kwargs)


# Legacy support
ColabEnvInitializer = ColabInitializer

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
        
        # Set default config
        self.config = self.get_default_config()
        self.logger.info(f"🛠️ ColabInitializer created for module: {module_name}")
        self.logger.debug("📋 Using in-memory configuration only (no persistence)")
    
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
        # Always use default config - no persistence
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
            ui_components = create_colab_ui_components(config=config, **kwargs)
            
            if not isinstance(ui_components, dict):
                raise ValueError("create_colab_ui_components() did not return a dictionary")
                
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
        """Setup module handlers
        
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
            if 'operation_container' in ui_components:
                try:
                    ui_components['operation_container'].log_error(error_msg)
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
            if 'operation_container' in ui_components:
                try:
                    ui_components['operation_container'].log_error(error_msg)
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
            if hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                try:
                    self._ui_components['operation_container'].log_error(error_msg)
                except:
                    pass
                    
            self._operation_handlers = {}
            
            # Try to log to operation container if available
            if hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                try:
                    self._ui_components['operation_container'].log_error(error_msg)
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
        """Implementation using existing ColabInitializer
        
        Returns:
            Dictionary containing UI components and initialization results
        """
        # Initialize and get UI components
        result = self._colab_initializer.initialize(**kwargs)
        
        # Ensure the result has the expected structure
        if not isinstance(result, dict):
            self.logger.error("❌ Invalid initialization result format")
            return {'success': False, 'error': 'Invalid initialization result format'}
            
        # Get the main container
        main_container = result.get('ui')
        if not main_container:
            self.logger.error("❌ No main container found in initialization result")
            return {'success': False, 'error': 'No main container found'}
            
        # Return the result with UI components
        return {
            'success': True,
            'ui': main_container,
            'ui_components': result.get('ui_components', {}),
            'operation_container': result.get('operation_container'),
            'module_handler': result.get('module_handler'),
            'config': result.get('config', {})
        }

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

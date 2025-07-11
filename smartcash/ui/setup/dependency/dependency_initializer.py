"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency module initializer following downloader pattern with requirements.txt installation
"""

from typing import Dict, Any
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.setup.dependency.components.dependency_ui import create_dependency_ui
from smartcash.ui.setup.dependency.configs.dependency_config_handler import DependencyConfigHandler
from smartcash.ui.setup.dependency.operations.operation_manager import DependencyOperationManager
from smartcash.ui.core.errors.handlers import create_error_response

class DependencyInitializer(ModuleInitializer):
    """Dependency initializer with complete UI and backend service integration
    
    Provides a structured approach to initializing the dependency module with
    proper error handling, logging, and UI component management. Follows the same
    initialization flow as the downloader with additional dependency-specific
    functionality including requirements.txt installation support.
    """

    def __init__(self):
        super().__init__(
            module_name='dependency',
            parent_module='setup',
            config_handler_class=DependencyConfigHandler
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default dependency configuration"""
        from .configs.dependency_defaults import get_default_dependency_config
        return get_default_dependency_config()
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create dependency UI components following the new pattern

        Args:
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Dictionary of UI components with 'ui_components' key
        """
        try:
            self.logger.info("🔧 Creating dependency UI with new pattern")
            
            # Create UI components
            ui_result = create_dependency_ui(config, **kwargs)
            
            # Check if create_dependency_ui returned None or an error occurred
            if ui_result is None:
                error_msg = "Failed to create dependency UI: create_dependency_ui returned None"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Ensure ui_components has the required structure
            if not isinstance(ui_result, dict):
                error_msg = f"Expected create_dependency_ui to return a dict, got {type(ui_result).__name__}"
                self.logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Handle case where ui_components is nested under 'ui_components' key
            if 'ui_components' in ui_result:
                ui_components = ui_result
            else:
                ui_components = {'ui_components': ui_result}
            
            # Ensure ui_components['ui_components'] is a dictionary
            if not isinstance(ui_components['ui_components'], dict):
                ui_components['ui_components'] = {}
            
            # Add metadata
            ui_components.update({
                'dependency_initialized': True,
                'module_name': 'dependency',
                'parent_module': 'setup',
                'config': config,
                'env': env or {}
            })
            
            # Ensure we have the required containers structure
            if 'containers' not in ui_components['ui_components']:
                ui_components['ui_components']['containers'] = {}
            
            try:
                # Set up operation manager with UI components
                self._setup_operation_manager(ui_components, config)
                
                # Initialize the operation manager if it exists
                if hasattr(self, 'operation_manager') and self.operation_manager:
                    if hasattr(self.operation_manager, 'initialize'):
                        self.operation_manager.initialize()
                
                self.logger.info("✅ UI components created successfully")
                return ui_components
                
            except Exception as e:
                self.logger.error(f"❌ Failed to set up operation manager: {e}")
                # Return the UI components even if operation manager setup fails
                return ui_components

        except Exception as e:
            self.logger.error(f"❌ Failed to create dependency UI components: {e}")
            import traceback
            self.logger.error(f"Error details: {traceback.format_exc()}")
            # Return a minimal UI with error message
            return {
                'ui_components': {
                    'containers': {},
                    'widgets': {}
                },
                'error': str(e),
                'dependency_initialized': False
            }

    def _setup_operation_manager(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Set up the operation manager for dependency operations.

        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
        """
        try:
            # Set up logger to use operation container's log accordion
            self._setup_logger_for_operation_container(ui_components)
            
            # Create operation manager
            operation_manager = DependencyOperationManager(
                config=config,
                ui_components=ui_components,
                logger=self.logger  # Pass the configured logger
            )
            
            # Store in UI components
            ui_components['operation_manager'] = operation_manager
            
            # Initialize the operation manager
            operation_manager.initialize()
            
            # Connect operation manager to UI components
            if 'widgets' in ui_components:
                # Connect install button
                if 'install_button' in ui_components['widgets']:
                    ui_components['widgets']['install_button'].on_click(
                        lambda b: operation_manager.execute_install(
                            operation_manager._get_selected_packages()
                        )
                    )
                
                # Connect update button
                if 'update_button' in ui_components['widgets']:
                    ui_components['widgets']['update_button'].on_click(
                        lambda b: operation_manager.execute_update(
                            operation_manager._get_selected_packages()
                        )
                    )
                
                # Connect uninstall button
                if 'uninstall_button' in ui_components['widgets']:
                    ui_components['widgets']['uninstall_button'].on_click(
                        lambda b: operation_manager.execute_uninstall(
                            operation_manager._get_selected_packages()
                        )
                    )
                
            self.logger.info("✅ Operation manager setup complete")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to set up operation manager: {e}")
            import traceback
            self.logger.error(f"Error details: {traceback.format_exc()}")
            raise
            
    def _setup_logger_for_operation_container(self, ui_components: Dict[str, Any]) -> None:
        """Set up logger to use operation container's log accordion.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Check if operation container exists and has log_message method
            if 'operation_container' in ui_components and hasattr(ui_components['operation_container'], 'log_message'):
                # Store original logger methods
                original_info = self.logger.info
                original_warning = self.logger.warning
                original_error = self.logger.error
                original_debug = self.logger.debug
                
                # Override logger methods to use operation container's log_message
                def log_to_operation_container(message, level='INFO'):
                    # Log to operation container
                    try:
                        ui_components['operation_container'].log_message(message, level)
                    except Exception:
                        pass  # Fallback silently if operation container logging fails
                    
                    # Also log to original logger based on level
                    if level == 'INFO':
                        original_info(message)
                    elif level == 'WARNING':
                        original_warning(message)
                    elif level == 'ERROR':
                        original_error(message)
                    elif level == 'DEBUG':
                        original_debug(message)
                
                # Replace logger methods
                self.logger.info = lambda msg: log_to_operation_container(msg, 'INFO')
                self.logger.warning = lambda msg: log_to_operation_container(msg, 'WARNING')
                self.logger.error = lambda msg: log_to_operation_container(msg, 'ERROR')
                self.logger.debug = lambda msg: log_to_operation_container(msg, 'DEBUG')
                
                # Log a test message to verify setup
                self.logger.info("📋 Logger connected to operation container")
                
        except Exception as e:
            # Use original logger to report error
            import logging
            logging.getLogger(self.__class__.__name__).error(f"Failed to set up logger for operation container: {e}")

    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup handlers following downloader pattern with OperationHandler

        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Updated UI components with handlers
        """
        try:
            # No changes needed here
            return ui_components
        except Exception as e:
            self.handle_error(f"Failed to setup module handlers: {str(e)}", exc_info=True)
            return ui_components

    def pre_initialize_checks(self, **kwargs) -> None:
        """Pre-initialization checks
        
        Args:
            **kwargs: Additional arguments
        """
        # Check if we're in a supported environment
        try:
            import IPython  # noqa: F401
            # Additional checks can be added here
        except ImportError:
            raise RuntimeError("Dependency management requires IPython environment")

    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implementation of the abstract _initialize_impl method.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        # Pre-initialization checks
        self.pre_initialize_checks(**kwargs)
        
        # Load configuration
        config = kwargs.get('config')
        if config is None:
            config = self._get_default_config()
        
        # Create UI components
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['config', 'env']}
        ui_components = self._create_ui_components(config, env=kwargs.get('env'), **filtered_kwargs)
        
        # Setup module handlers
        ui_components = self._setup_module_handlers(ui_components, config, **filtered_kwargs)
        
        # Post-initialization cleanup
        self.post_initialize_cleanup()
        
        return ui_components
    
# Global instance and public API
_dependency_initializer = DependencyInitializer()

class DependencyDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for dependency module"""
    
    def __init__(self):
        super().__init__(module_name="dependency", parent_module="setup")
        self._dependency_initializer = DependencyInitializer()
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation using existing DependencyInitializer"""
        return self._dependency_initializer._initialize_impl(**kwargs)

# Global display initializer instance
_dependency_display_initializer = DependencyDisplayInitializer()

def initialize_dependency_ui(env=None, config=None, **kwargs) -> None:
    """Initialize and display dependency UI using DisplayInitializer

    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    
    Note:
        This function displays the UI directly and returns None.
        Use get_dependency_components() if you need access to the components dictionary.
    """
    _dependency_display_initializer.initialize_and_display(config=config, env=env, **kwargs)

def get_dependency_components(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Get dependency components dictionary without displaying UI

    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components
    """
    return _dependency_initializer.initialize(config=config, env=env, **kwargs)

def display_dependency_ui(env=None, config=None, **kwargs) -> None:
    """Display dependency UI (alias for initialize_dependency_ui)

    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_dependency_ui(env=env, config=config, **kwargs)

def get_dependency_initializer() -> DependencyInitializer:
    """Get or create dependency initializer instance.
    
    Returns:
        DependencyInitializer instance
    """
    global _dependency_initializer
    
    if _dependency_initializer is None:
        _dependency_initializer = DependencyInitializer()
    
    return _dependency_initializer

# Legacy support
DependencyEnvInitializer = DependencyInitializer
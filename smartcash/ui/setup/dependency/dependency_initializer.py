"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency module initializer following downloader pattern with requirements.txt installation
"""

from typing import Dict, Any
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.setup.dependency.components.dependency_ui import create_dependency_ui_components
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
        """Create dependency UI components following downloader pattern

        Args:
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating dependency UI components with operation container")
            ui_components = create_dependency_ui_components(config, **kwargs)

            # Add metadata following the established pattern
            ui_components.update({
                'dependency_initialized': True,
                'module_name': 'dependency',
                'logger': self.logger,
                'config': config,
                'env': env
            })

            self.logger.info(f"✅ UI components created successfully: {len(ui_components)} components")
            return ui_components
        except Exception as e:
            self.handle_error(f"Failed to create UI components: {str(e)}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI dependency")

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
            self.logger.info("🔧 Setting up dependency operation manager")

            # Get operation container from UI components and set up operation handler
            operation_container = ui_components.get('operation_manager')
            if operation_container:
                # Create operation manager with operation container
                operation_manager = DependencyOperationManager(
                    config=config,
                    operation_container=operation_container
                )
                
                # Store UI components reference in operation manager
                operation_manager._ui_components = ui_components
                
                # Initialize the operation manager
                operation_manager.initialize()
                
                # Store operation manager in UI components
                ui_components['dependency_operation_manager'] = operation_manager
                
                self.logger.info("✅ Operation manager setup complete")
            else:
                self.logger.warning("⚠️ No operation container found in UI components")

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
"""
Dependency Initializer Module.

This module provides the DependencyInitializer class which is responsible for
initializing and managing the dependency management UI and its handlers.

It handles the complete lifecycle of the dependency management interface,
including UI component creation, handler setup, and configuration management.
"""

# Standard library imports
import sys
from typing import Dict, Any, Optional
import ipywidgets as widgets

# Absolute imports
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.common.logger import get_logger
from smartcash.ui.setup.dependency.utils.core.validators import DependencyValidator
from smartcash.ui.setup.dependency.utils.types import UIComponents
from smartcash.ui.utils.logging_utils import suppress_all_outputs, restore_stdout
# Global instance to avoid circular imports
_dependency_initializer: Optional['DependencyInitializer'] = None

class DependencyInitializer(CommonInitializer):
    """Initializes and manages the dependency management UI and handlers.
    
    This class is responsible for:
    - Creating and configuring UI components
    - Setting up event handlers
    - Managing the dependency lifecycle
    - Coordinating between different parts of the dependency system
    """
    
    def __init__(self, logger=None, config_provider=None):
        """Initialize the dependency initializer.
        
        Args:
            logger: Custom logger instance. If None, a default logger will be used.
            config_provider: Function that returns the default config. If None, the default
                           implementation will be used.
        """
        from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
        super().__init__('dependency', DependencyConfigHandler)
        self.logger = logger or get_logger('smartcash.ui.setup.dependency.initializer')
        self._config_provider = config_provider or self._get_default_config_impl
    
    def _get_default_config_impl(self) -> Dict[str, Any]:
        """Default implementation of config provider.
        
        Returns:
            Dict[str, Any]: Default dependency configuration
        """
        from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
        try:
            # Create a temporary handler instance to get default config
            handler = DependencyConfigHandler()
            return handler.get_default_dependency_config()
        except Exception as e:
            self.logger.warning(f"Failed to get default config from handler: {str(e)}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Return fallback configuration when default config cannot be loaded.
        
        Returns:
            Dict[str, Any]: Fallback configuration dictionary
        """
        return {
            'module_name': 'dependency',
            'dependencies': {
                'torch': {'version': 'latest', 'required': True},
                'torchvision': {'version': 'latest', 'required': True}, 
                'ultralytics': {'version': 'latest', 'required': True}
            },
            'auto_update': True,
            'check_on_startup': True,
            'run_analysis_on_startup': True
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default dependency configuration.
        
        This method implements the abstract method from CommonInitializer.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            return self._config_provider()
        except Exception as e:
            self.logger.warning(f"Could not load default config: {str(e)}")
            return self._get_fallback_config()
    
    # _get_default_config method is now defined above as part of __init__
    
    def _get_ui_root(self, ui_components: UIComponents) -> 'UIComponent':
        """Get the root UI component from components dictionary.
        
        Args:
            ui_components: Dictionary containing UI components
            
        Returns:
            The root UI component
            
        Raises:
            KeyError: If the UI root component is not found
        """
        if 'ui' not in ui_components:
            raise KeyError("Root UI component 'ui' not found in components")
        return ui_components['ui']
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> UIComponents:
        """Create and return UI components for dependency management.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing the created UI components
            
        Raises:
            RuntimeError: If UI component creation fails
        """
        try:
            from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
            components = create_dependency_main_ui(config)
            self.logger.debug("Successfully created dependency UI components")
            return components
        except ImportError as e:
            error_msg = f"Failed to import UI components: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to create dependency UI components: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _setup_handlers(self, ui_components: UIComponents, config: Dict[str, Any], **kwargs) -> UIComponents:
        """Setup event handlers for dependency management.
        
        Args:
            ui_components: Dictionary containing UI components
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of UI components with handlers attached
            
        Raises:
            RuntimeError: If handler setup fails
        """
        try:
            from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
            
            # Setup handlers and get the handler map
            handlers = setup_dependency_handlers(ui_components)
            
            # Store handlers in ui_components for later access
            ui_components['handlers'] = handlers
            
            self.logger.info("Successfully set up dependency handlers")
            return ui_components
            
        except ImportError as e:
            error_msg = f"Failed to import dependency handlers: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to setup dependency handlers: {str(e)}"
            self.logger.error(f"{error_msg}\n{str(e)}")
            raise RuntimeError(error_msg) from e
    
    def _after_init_checks(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Run post-initialization checks and update UI accordingly.
        
        This method runs after all components and handlers are set up.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            
        Raises:
            RuntimeError: If post-initialization checks fail
        """
        try:
            # Analyze packages and update status
            from smartcash.ui.setup.dependency.utils.package.status import analyze_packages
            
            # Create validator instance and validate config
            validator = DependencyValidator()
            validation_result = validator.validate_config(config)
            
            if not validation_result['valid']:
                error_msg = f"Invalid configuration: {', '.join(validation_result['issues'])}"
                raise ValueError(error_msg)
            
            # Analyze packages and get status
            package_status = analyze_packages(config)
            
            # Update UI based on package status
            if 'status_panel' in ui_components:
                status_text = "\n".join(
                    f"{pkg}: {'✓' if status.get('installed', False) else '✗'}"
                    for pkg, status in package_status.items()
                )
                ui_components['status_panel'].value = status_text
                
        except Exception as e:
            error_msg = f"Post-initialization check failed: {str(e)}"
            # Log the error with traceback
            self.logger.error(f"{error_msg}\n{str(e.__traceback__)}")
            
            # Update UI with error message
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = f"Error during initialization: {str(e)}"
                
            raise RuntimeError(error_msg) from e
    
    def _after_init_checks(self, **kwargs) -> None:
        """Run pre-initialization checks for dependencies including package analysis.
        
        This method performs the following tasks:
        1. Validates required system dependencies
        2. Runs package analysis if enabled in config
        3. Updates UI with any warnings or errors
        
        Args:
            **kwargs: Additional keyword arguments including:
                - ui_components: Dictionary of UI components
                - config: Configuration dictionary (optional)
        """
        ui_components = kwargs.get('ui_components', {})
        config = kwargs.get('config', {})
        
        # Skip if no UI components provided
        if not ui_components:
            self.logger.warning("No UI components provided for pre-initialization checks")
            return
            
        # Check if package analysis is enabled in config
        run_analysis = config.get('run_analysis_on_startup', True)
        if not run_analysis:
            self.logger.debug("Skipping package analysis (disabled in config)")
            return
            
        try:
            self.logger.info("Running package analysis as part of initialization...")
            
            # Import analysis handler
            from .handlers.analysis_handler import AnalysisHandler
            
            # Initialize and run analysis
            analysis_handler = AnalysisHandler(ui_components)
            analysis_handler.run_analysis()
            
            self.logger.info("Package analysis completed during initialization")
            
        except ImportError as e:
            error_msg = f"Failed to import analysis handler: {str(e)}"
            self.logger.error(error_msg)
            self._update_status_panel(ui_components, error_msg, 'error')
            
    def initialize_ui(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the dependency UI components.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing the initialized UI components
            
        Raises:
            RuntimeError: If UI initialization fails
        """
        try:
            # Create UI components first
            ui_components = self._create_ui_components(config)
            
            # Setup handlers
            ui_components = self._setup_handlers(ui_components, config)
            
            # Run post-initialization checks after components and handlers are set up
            self._after_init_checks(ui_components=ui_components, config=config)
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Failed to initialize dependency UI: {str(e)}"
            # Log the error with the exception details
            self.logger.error(f"{error_msg}\n{str(e)}")
            raise RuntimeError(error_msg) from e
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the dependency management system.
        
        This is the main entry point for initializing the dependency management system.
        It handles the complete setup process including UI creation, handler setup,
        and error handling.
        
        Args:
            config: Optional configuration dictionary. If not provided, defaults will be used.
            
        Returns:
            Dictionary containing the initialized UI components and handlers
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Use provided config or get default
            if config is None:
                config = self._get_default_config()
            
            # Initialize UI components
            ui_components = self.initialize_ui(config)
            
            # Setup handlers
            handlers = self._setup_handlers(ui_components, config)
            
            # Run post-initialization checks
            self._after_init_checks(ui_components=ui_components, config=config)
            
            # Return both UI components and handlers
            return {
                'ui': ui_components,
                'handlers': handlers
            }
            
        except Exception as e:
            error_msg = f"Failed to initialize dependency management: {str(e)}"
            # Log the error with the exception details
            self.logger.error(f"{error_msg}\n{str(e)}")
            raise RuntimeError(error_msg) from e
    
    def _update_status_panel(self, ui_components: UIComponents, message: str, 
                           level: str = 'info') -> None:
        """Update the status panel with a message.
        
        Args:
            ui_components: Dictionary of UI components
            message: Message to display
            level: Message level ('info', 'warning', 'error')
        """
        try:
            from smartcash.ui.setup.dependency.utils.ui.state import update_status_panel
            update_status_panel(ui_components, message, level)
        except ImportError:
            self.logger.warning(
                f"Could not update status panel: {message} (level: {level})"
            )

# Module-level variable to store the singleton instance
_dependency_initializer = None

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None, display_ui: bool = True) -> Dict[str, Any]:
    """Initialize and optionally display the dependency UI.
    
    This is the main entry point for initializing the dependency management UI.
    It handles the complete setup process including UI creation, handler setup,
    and error handling.
    
    Args:
        config: Optional configuration dictionary. If not provided, defaults will be used.
        display_ui: Whether to automatically display the UI components. Defaults to True.
                  Set to False if you want to handle display manually.
        
    Returns:
        Dictionary containing the initialized UI components. If an error occurs,
        returns a dictionary with error information.
    """
    global _dependency_initializer
    
    try:
        # Initialize the singleton instance if it doesn't exist
        if _dependency_initializer is None:
            _dependency_initializer = DependencyInitializer()
        
        # Ensure required config fields are present
        if config is None:
            config = {}
            
        # Set default values for required fields if not provided
        if 'dependencies' not in config:
            # Initialize as empty dict since that's what the validator expects
            config['dependencies'] = {}
        elif isinstance(config['dependencies'], list):
            # Convert list to dict with package names as keys if needed
            config['dependencies'] = {dep: {} for dep in config['dependencies']}
            
        if 'install_options' not in config:
            config['install_options'] = {
                'run_analysis_on_startup': True,
                'auto_install': False,
                'verbose': True
            }
        
        # Initialize the UI with the config
        ui_components = _dependency_initializer.initialize_ui(config)
        
        # Display the UI if requested
        if display_ui:
            from IPython.display import display
            
        if 'container' in ui_components:
            return ui_components['container']
        elif 'ui' in ui_components:
            return ui_components['ui']
        else:
            raise ValueError("No valid UI container found in components")
        
    except Exception as e:
        restore_stdout()  # Ensure output is restored even on error
        error_fallback = _create_error_fallback(str(e))
        if 'container' in error_fallback:
            return error_fallback['container']
        return error_fallback

def _create_error_fallback(error_message: str, traceback: Optional[str] = None) -> widgets.VBox:
    """Create a fallback UI component to display error messages."""
    from smartcash.ui.components import create_error_component
    return create_error_component("Initialization Error", error_message, traceback)

__all__ = ['initialize_dependency_ui']
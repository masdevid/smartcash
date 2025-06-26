# File: smartcash/ui/setup/dependency/dependency_initializer.py
# Desc: Dependency management UI and handler initialization

# Standard library imports
import traceback
from typing import Dict, Any, Optional, Type

# Absolute imports
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.utils.core.validators import DependencyValidator
from smartcash.ui.setup.dependency.utils.types import UIComponents
from smartcash.ui.utils.logging_utils import restore_stdout, suppress_all_outputs

class DependencyInitializer(CommonInitializer):
    """Manages dependency UI components and their lifecycle."""
    
    def __init__(self, logger=None, config_provider=None):
        """Initialize the dependency initializer.
        
        Args:
            logger: Custom logger instance. If None, a default logger will be used.
            config_provider: Function that returns the default config. If None, the default implementation will be used.
        """
        from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
        super().__init__('setup.dependency', DependencyConfigHandler)
        self._config_provider = config_provider or self._get_default_config
        
        # Override logger if custom one is provided
        if logger is not None:
            self.logger = logger
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default dependency configuration.
        
        Returns:
            Dict containing the default dependency configuration from the config handler.
        """
        return self.config_handler.get_default_config()
    
    def _get_ui_root(self, ui_components: Dict[str, Any]) -> Any:
        """Get the root UI component from components dictionary.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            The root UI component or None if not found
        """
        return ui_components.get('ui')
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create and return UI components for dependency management.
        
        Args:
            config: Loaded configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        components = create_dependency_main_ui(config)
        self.logger.debug("Successfully created dependency UI components")
        return components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup event handlers for dependency management.
        
        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
            **kwargs: Additional arguments
            
        Returns:
            Updated dictionary of UI components with handlers
        """
        from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
        
        # Setup handlers and get the handler map
        handlers = setup_dependency_handlers(ui_components)
        
        # Store handlers in ui_components for later access
        ui_components['handlers'] = handlers
        self.logger.info("Successfully set up dependency handlers")
        return ui_components
    
    def _after_init_checks(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> None:
        """Run post-initialization checks and update UI accordingly."""
        # Validate config
        validator = DependencyValidator()
        validation_result = validator.validate_config(config)
        
        if not validation_result['valid']:
            error_msg = f"Invalid configuration: {', '.join(validation_result['issues'])}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Run package analysis if enabled
        if config.get('install_options', {}).get('run_analysis_on_startup', True):
            self._run_package_analysis(ui_components, config)
    
    def _run_package_analysis(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Run package analysis and update UI with results."""
        try:
            from smartcash.ui.setup.dependency.utils.package.status import analyze_packages
            
            self.logger.info("Running package analysis...")
            package_status = analyze_packages(config)
            
            # Update UI with package status
            if 'status_panel' in ui_components:
                status_text = "\n".join(
                    f"{pkg}: {'✓' if status.get('installed', False) else '✗'}"
                    for pkg, status in package_status.items()
                )
                ui_components['status_panel'].value = status_text
                
        except ImportError as e:
            error_msg = f"Failed to import analysis module: {str(e)}"
            self.logger.error(error_msg)
            self._update_status_panel(ui_components, error_msg, 'error')
        except Exception as e:
            error_msg = f"Package analysis failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self._update_status_panel(ui_components, error_msg, 'error')
    
    def _update_status_panel(self, ui_components: UIComponents, message: str, level: str = 'info') -> None:
        """Update the status panel with a message."""
        try:
            from smartcash.ui.setup.dependency.utils.ui.state import update_status_panel
            update_status_panel(ui_components, message, level)
        except ImportError:
            self.logger.warning(f"Could not update status panel: {message} (level: {level})")

# Module-level variable to store the singleton instance
_dependency_initializer = None

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None, display_ui: bool = True) -> Optional[Dict[str, Any]]:
    """
    Initialize and optionally display the dependency UI.
    
    This is the main entry point for initializing the dependency management UI.
    It handles the complete setup process including UI creation, handler setup,
    and error handling.
    
    Args:
        config: Optional configuration dictionary. If None, default config will be used.
        display_ui: Whether to display the UI (default: True)
        
    Returns:
        Dictionary containing UI components and handlers
        
    Raises:
        Exception: If initialization fails, the exception will propagate up
    """
    global _dependency_initializer
    
    # Suppress outputs during initialization
    suppress_all_outputs()
    
    try:
        from IPython.display import display
        from ipywidgets import Output
        
        # Create a new initializer if one doesn't exist
        if _dependency_initializer is None:
            _dependency_initializer = DependencyInitializer()
        
        # Ensure config is properly formatted
        if config is None:
            config = _dependency_initializer._get_default_config()
        else:
            # Convert list of dependencies to dict if needed
            if 'dependencies' in config and isinstance(config['dependencies'], list):
                config['dependencies'] = {dep: {} for dep in config['dependencies']}
        
        # Initialize UI components and handlers
        ui_components = _dependency_initializer.initialize(config)
        
        # Clear any existing output before displaying
        if display_ui and 'ui' in ui_components and ui_components['ui'] is not None:
            # Clear any existing output
            from IPython.display import clear_output
            clear_output(wait=True)
            
            # Create a clean output widget to prevent double display
            output = Output()
            with output:
                display(ui_components['ui'])
            
            # Store the output widget in the components
            ui_components['_output'] = output
            
            # Display the output widget
            display(output)
        
        return ui_components
        
    except Exception as e:
        # Log the error and re-raise
        error_msg = f"Failed to initialize dependency UI: {str(e)}"
        if _dependency_initializer and hasattr(_dependency_initializer, 'logger'):
            _dependency_initializer.logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
        
    finally:
        # Always restore output
        restore_stdout()

__all__ = ['initialize_dependency_ui']
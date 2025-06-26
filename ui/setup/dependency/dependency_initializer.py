"""
Dependency Initializer Module

This module provides the initialization logic for the dependency management UI,
handling the setup of UI components, configuration, and event handlers.
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
from smartcash.ui.setup.dependency.utils.core.validators import DependencyValidator

class DependencyInitializer(CommonInitializer):
    """Initializer for the dependency management module."""
    
    def __init__(self):
        super().__init__(
            module_name='setup.dependency',
            config_handler_class=DependencyConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components for dependency management.
        
        Args:
            config: Configuration for UI initialization
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing UI components
        """
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        
        ui_components = create_dependency_main_ui(config)
        
        if not isinstance(ui_components, dict):
            raise ValueError("UI components must be a dictionary")
            
        if not ui_components:
            raise ValueError("UI components cannot be empty")
            
        ui_components.update({
            'module_name': self.module_name,
            'config_handler': self.config_handler
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Set up event handlers for the UI components.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration used for setup
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components with handlers
        """
        from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
        
        handlers = setup_dependency_handlers(ui_components)
        ui_components['handlers'] = handlers
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
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration for the dependency management module.
        
        Returns:
            Dictionary containing default configuration values from the config handler
        """
        if hasattr(self, 'config_handler') and hasattr(self.config_handler, 'get_default_config'):
            return self.config_handler.get_default_config()
            
        # Fallback to a basic config if config_handler is not available
        return {
            'dependencies': {},
            'install_options': {
                'run_analysis_on_startup': True,
                'auto_install': False,
                'upgrade_strategy': 'if_needed'
            },
            'ui': {
                'show_advanced': False,
                'theme': 'light'
            },
            'module_name': self.module_name,
            'version': '1.0.0'
        }
    
    def _run_package_analysis(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Run package analysis if enabled in config."""
        if 'handlers' in ui_components and 'analyze_packages' in ui_components['handlers']:
            try:
                ui_components['handlers']['analyze_packages']()
            except Exception as e:
                self.logger.error(f"Failed to run package analysis: {str(e)}", exc_info=True)


def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Initialize the dependency management UI.
    
    Args:
        config: Optional configuration for initialization
        **kwargs: Additional arguments to pass to the initializer
        
    Returns:
        The main UI component
    """
    return DependencyInitializer().initialize(config=config, **kwargs)

__all__ = ['initialize_dependency_ui']
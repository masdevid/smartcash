"""
Optimized factory for creating and displaying Dependency UI module.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule

class DependencyUIFactory(UIFactory):
    """Optimized factory for Dependency UI module creation and display."""
    
    @classmethod
    def create_dependency_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DependencyUIModule:
        """Create and initialize DependencyUIModule with given configuration.
        
        Args:
            config: Optional configuration for the module
            **kwargs: Additional arguments for module initialization
                
        Returns:
            Initialized DependencyUIModule instance
        """
        module = DependencyUIModule()
        
        # Update config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
        
        # Initialize the module
        module.initialize()
        
        return module
    
    @classmethod
    def create_and_display_dependency(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Create and display the Dependency UI module.
        
        Args:
            config: Optional configuration for the module
            **kwargs: Additional arguments for module initialization
        """
        from smartcash.ui.core import ui_utils
        
        module = cls.create_dependency_module(config=config, **kwargs)
        ui_utils.display_ui_module(module=module, module_name="Dependency", **kwargs)


def create_dependency_display(**kwargs):
    """Create a display function for the dependency UI.
    
    Args:
        **kwargs: Configuration options for the dependency UI
        
    Returns:
        A callable that will display the dependency UI when called
    """
    def display_fn():
        DependencyUIFactory.create_and_display_dependency(**kwargs)
    
    return display_fn

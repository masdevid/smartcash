"""
Optimized factory for creating and displaying Dependency UI module.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function

class DependencyUIFactory(UIFactory):
    """
    Optimized factory for Dependency UI module creation and display.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> DependencyUIModule:
        """
        Create a new instance of DependencyUIModule.
        
        Args:
            config: Optional configuration for the module
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New DependencyUIModule instance
        """
        module = DependencyUIModule()
        
        # Update config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_dependency_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> DependencyUIModule:
        """Create and initialize DependencyUIModule with caching.
        
        Args:
            config: Optional configuration for the module
            force_refresh: Force refresh cache if True
            **kwargs: Additional arguments for module initialization
                
        Returns:
            Initialized DependencyUIModule instance with caching
        """
        return create_ui_factory_method(
            module_class=DependencyUIModule,
            module_name="Dependency",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
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
                - auto_display: Boolean, whether to automatically display the UI (default: True)
                - force_refresh: Boolean, whether to force refresh the cache (default: False)
        """
        display_fn = create_display_function(
            factory_class=cls,
            create_method_name='create_dependency_module',
            module_name='Dependency',
            config=config,
            **kwargs
        )
        return display_fn()


def create_dependency_display(**kwargs):
    """Create a display function for the dependency UI.
    
    Args:
        **kwargs: Configuration options for the dependency UI
            - config: Optional configuration for the module
            - auto_display: Boolean, whether to automatically display the UI (default: True)
            - force_refresh: Boolean, whether to force refresh the cache (default: False)
            
    Returns:
        A callable that will display the dependency UI when called
    """
    def display_fn():
        DependencyUIFactory.create_and_display_dependency(**kwargs)
    
    return display_fn

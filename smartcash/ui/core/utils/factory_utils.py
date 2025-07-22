"""
Utility functions for UI Factory pattern implementation.

This module provides common functionality for implementing the UI Factory pattern
with consistent caching and lifecycle management.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/core/utils/factory_utils.py
"""

from typing import Any, Dict, Optional, Type, TypeVar, Callable
from functools import wraps
import logging

T = TypeVar('T')

def create_ui_factory_method(
    module_class: Type[T],
    module_name: str,
    create_module_func: Callable[..., T]
) -> Callable[..., T]:
    """
    Create a factory method for UI modules with standardized caching.
    
    Args:
        module_class: The UI module class type
        module_name: Name of the module for logging
        create_module_func: Function that creates a new module instance
        
    Returns:
        A factory method with caching and error handling
    """
    logger = logging.getLogger(__name__)
    
    @wraps(create_module_func)
    def factory_method(
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> T:
        """
        Create and cache a UI module instance.
        
        Args:
            config: Optional configuration dictionary
            force_refresh: If True, bypass cache and create new instance
            **kwargs: Additional arguments for module creation
            
        Returns:
            An initialized module instance
        """
        from smartcash.ui.core.ui_factory import UIFactory
        
        cache_key = f"{module_name.lower()}_module_{hash(str(config))}"
        
        try:
            # Check cache first if not forcing refresh
            if not force_refresh:
                cached_module = UIFactory._get_cached_module(cache_key)
                if cached_module is not None:
                    if hasattr(cached_module, 'log_debug'):
                        cached_module.log_debug(f"✅ Using cached {module_name} module instance")
                    else:
                        logger.debug("✅ Using cached %s module instance", module_name)
                    return cached_module
            
            # Create new instance
            module = create_module_func(config=config, **kwargs)
            
            # Initialize with validation
            if hasattr(module, 'initialize'):
                initialization_result = module.initialize()
                if not initialization_result:
                    UIFactory._invalidate_cache(cache_key)
                    raise RuntimeError(f"{module_name} module initialization failed")
            
            # Cache the module instance
            UIFactory._cache_module(cache_key, module)
            
            # Log success
            if hasattr(module, 'log_debug'):
                module.log_debug(f"✅ Successfully initialized {module_name} module")
            else:
                logger.debug("✅ Successfully initialized %s module", module_name)
                
            return module
            
        except Exception as e:
            # Invalidate cache on error
            UIFactory._invalidate_cache(cache_key)
            
            # Log error
            error_msg = f"Failed to create {module_name} module: {e}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise
    
    return factory_method

def create_display_function(
    factory_class,
    create_method_name: str,
    module_name: str,
    **default_kwargs
) -> Callable[..., None]:
    """
    Create a display function for a UI module.
    
    Args:
        factory_class: The factory class containing the create method
        create_method_name: Name of the create method in the factory class
        module_name: Display name of the module
        **default_kwargs: Default arguments for the create method
        
    Returns:
        A function that creates and displays the UI module
    """
    from smartcash.ui.core import ui_utils
    
    def display_function(**kwargs):
        """Create and display the UI module."""
        # Merge default kwargs with provided kwargs
        merged_kwargs = {**default_kwargs, **kwargs}
        
        # Get the create method
        create_method = getattr(factory_class, create_method_name)
        
        # Create the module
        module = create_method(**merged_kwargs)
        
        # Only pass clear_output to display_ui, not the entire config
        display_kwargs = {
            'clear_output': merged_kwargs.get('clear_output', True)
        }
        
        ui_utils.display_ui_module(
            module=module,
            module_name=module_name,
            **display_kwargs
        )
        
        return None
    
    # Update docstring
    display_function.__doc__ = f"""
    Create and display the {module_name} UI.
    
    Args:
        **kwargs: Additional arguments for {module_name} module creation
            - config: Optional configuration dictionary
            - force_refresh: Boolean, whether to force refresh the cache (default: False)
            - auto_display: Boolean, whether to automatically display the UI (default: True)
            
    Returns:
        None (displays the UI using IPython.display)
    """
    
    return display_function

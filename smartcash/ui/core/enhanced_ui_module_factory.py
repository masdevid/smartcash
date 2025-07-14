"""
Enhanced UI Module Factory with standardized initialization and display patterns.

This factory provides consistent creation and display logic for all UI modules.
"""

from typing import Dict, Any, Optional, Callable, Type
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.utils.display_utils import safe_display
from smartcash.ui.logger import get_module_logger


class EnhancedUIModuleFactory:
    """
    Factory for creating and displaying UI modules with standardized patterns.
    
    This factory eliminates the need for individual modules to implement
    their own initialization and display logic.
    """
    
    @staticmethod
    def create_and_display(
        module_class: Type[BaseUIModule],
        config: Optional[Dict[str, Any]] = None,
        display: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Create and optionally display a UI module.
        
        Args:
            module_class: UI module class (must inherit from BaseUIModule)
            config: Optional configuration dictionary
            display: Whether to display the UI immediately
            **kwargs: Additional arguments for module creation
            
        Returns:
            Module information if display=False, None if display=True
        """
        logger = get_module_logger("smartcash.ui.core.enhanced_factory")
        
        try:
            # Create module instance
            module = module_class(**kwargs)
            
            # Initialize with config
            if config:
                module._initialize_config_handler(config)
            
            # Initialize the module
            if not module.initialize():
                raise RuntimeError(f"Failed to initialize {module.full_module_name}")
            
            if display:
                # Display the UI
                display_result = module.display_ui()
                
                if display_result and not display_result.get('success', False):
                    logger.error(f"Failed to display {module.full_module_name}: {display_result.get('message', 'Unknown error')}")
                    
                    # Show error in UI
                    from smartcash.ui.core.errors.error_component import create_error_component
                    error_component = create_error_component(
                        error_message=display_result.get('message', 'Failed to display UI'),
                        title=f"🚨 {module.module_name.title()} Display Error",
                        error_type="error"
                    )
                    
                    if error_component and 'widget' in error_component:
                        safe_display(error_component['widget'])
                
                return None
            else:
                # Return module information
                return module.get_module_info()
                
        except Exception as e:
            error_msg = f"Failed to create/display module: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if display:
                # Show error in UI
                from smartcash.ui.core.errors.error_component import create_error_component
                error_component = create_error_component(
                    error_message=error_msg,
                    title="🚨 Module Creation Error",
                    error_type="error"
                )
                
                if error_component and 'widget' in error_component:
                    safe_display(error_component['widget'])
            
            return {'error': error_msg} if not display else None
    
    @staticmethod
    def create_display_function(
        module_class: Type[BaseUIModule],
        function_name: str = None
    ) -> Callable:
        """
        Create a standardized display function for a UI module.
        
        Args:
            module_class: UI module class
            function_name: Optional custom function name
            
        Returns:
            Display function that can be used in cell files
        """
        def display_function(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
            """
            Initialize and display the UI module.
            
            Args:
                config: Optional configuration dictionary
                **kwargs: Additional arguments for module creation
            """
            return EnhancedUIModuleFactory.create_and_display(
                module_class=module_class,
                config=config,
                display=True,
                **kwargs
            )
        
        # Set function name and docstring
        if function_name:
            display_function.__name__ = function_name
        else:
            display_function.__name__ = f"initialize_{module_class.__name__.lower().replace('uimodule', '')}_ui"
        
        display_function.__doc__ = f"""
        Initialize and display the {module_class.__name__} UI.
        
        This function creates and displays the UI module with consistent
        error handling and logging management.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments for module creation
        """
        
        return display_function
    
    @staticmethod
    def create_component_function(
        module_class: Type[BaseUIModule],
        function_name: str = None
    ) -> Callable:
        """
        Create a function that returns UI components without displaying.
        
        Args:
            module_class: UI module class
            function_name: Optional custom function name
            
        Returns:
            Component function that returns module information
        """
        def component_function(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
            """
            Create UI module and return component information.
            
            Args:
                config: Optional configuration dictionary
                **kwargs: Additional arguments for module creation
                
            Returns:
                Module information dictionary
            """
            result = EnhancedUIModuleFactory.create_and_display(
                module_class=module_class,
                config=config,
                display=False,
                **kwargs
            )
            
            return result or {'error': 'Failed to create module'}
        
        # Set function name and docstring
        if function_name:
            component_function.__name__ = function_name
        else:
            component_function.__name__ = f"get_{module_class.__name__.lower().replace('uimodule', '')}_components"
        
        component_function.__doc__ = f"""
        Create {module_class.__name__} and return UI components.
        
        This function creates the UI module and returns component information
        without displaying the UI.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments for module creation
            
        Returns:
            Dictionary containing module information and UI components
        """
        
        return component_function
    
    @staticmethod
    def create_legacy_wrapper(
        legacy_function: Callable,
        module_name: str,
        parent_module: str = None
    ) -> Callable:
        """
        Create a wrapper for legacy initialization functions.
        
        Args:
            legacy_function: Existing legacy function
            module_name: Module name
            parent_module: Parent module name
            
        Returns:
            Wrapped function with enhanced error handling
        """
        def wrapper(config: Optional[Dict[str, Any]] = None, display: bool = True, **kwargs):
            """
            Legacy function wrapper with enhanced error handling.
            
            Args:
                config: Optional configuration dictionary
                display: Whether to display the UI
                **kwargs: Additional arguments
                
            Returns:
                Result from legacy function
            """
            logger = get_module_logger(f"smartcash.ui.{parent_module}.{module_name}" if parent_module else f"smartcash.ui.{module_name}")
            
            try:
                # Use display initializer for consistent logging suppression
                if display:
                    display_initializer = DisplayInitializer(module_name, parent_module)
                    display_initializer.initialize_and_display(config=config, **kwargs)
                    return None
                else:
                    return legacy_function(config=config, display=False, **kwargs)
                    
            except Exception as e:
                error_msg = f"Legacy function failed for {module_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                if display:
                    # Show error in UI
                    from smartcash.ui.core.errors.error_component import create_error_component
                    error_component = create_error_component(
                        error_message=error_msg,
                        title=f"🚨 {module_name.title()} Error",
                        error_type="error"
                    )
                    
                    if error_component and 'widget' in error_component:
                        safe_display(error_component['widget'])
                
                return {'error': error_msg} if not display else None
        
        # Set function name and docstring
        wrapper.__name__ = f"initialize_{module_name}_ui"
        wrapper.__doc__ = f"""
        Initialize and display the {module_name} UI (legacy wrapper).
        
        This function wraps the legacy initialization function with enhanced
        error handling and consistent logging management.
        
        Args:
            config: Optional configuration dictionary
            display: Whether to display the UI immediately
            **kwargs: Additional keyword arguments
        """
        
        return wrapper


# Convenience functions
def create_display_function(module_class: Type[BaseUIModule], function_name: str = None) -> Callable:
    """
    Create a display function for a UI module.
    
    Args:
        module_class: UI module class
        function_name: Optional custom function name
        
    Returns:
        Display function
    """
    return EnhancedUIModuleFactory.create_display_function(module_class, function_name)


def create_component_function(module_class: Type[BaseUIModule], function_name: str = None) -> Callable:
    """
    Create a component function for a UI module.
    
    Args:
        module_class: UI module class
        function_name: Optional custom function name
        
    Returns:
        Component function
    """
    return EnhancedUIModuleFactory.create_component_function(module_class, function_name)


def create_legacy_wrapper(legacy_function: Callable, module_name: str, parent_module: str = None) -> Callable:
    """
    Create a wrapper for legacy initialization functions.
    
    Args:
        legacy_function: Existing legacy function
        module_name: Module name
        parent_module: Parent module name
        
    Returns:
        Wrapped function
    """
    return EnhancedUIModuleFactory.create_legacy_wrapper(legacy_function, module_name, parent_module)
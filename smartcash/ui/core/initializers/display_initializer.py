"""
File: smartcash/ui/core/initializers/display_initializer.py
Description: Display-capable initializer that extends base initializer with UI display functionality

This initializer provides:
1. Consistent UI display instead of returning dictionaries
2. Proper logging management (suppress early logs, restore after UI ready)
3. Beautiful error display using existing error_component.py
4. No code duplication across modules
"""

import logging
from typing import Dict, Any, Optional, Union
from IPython.display import display
from contextlib import contextmanager

from .base_initializer import BaseInitializer
from smartcash.ui.core.errors import (
    create_error_component
)


class DisplayInitializer(BaseInitializer):
    """
    Base initializer with UI display functionality.
    
    This class extends BaseInitializer to provide:
    - Automatic UI display instead of returning dictionaries
    - Logging suppression during initialization  
    - Beautiful error displays using core error components
    - Consistent behavior across all modules
    """
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._original_log_level = None
        self._original_smartcash_log_level = None
    
    @contextmanager
    def _suppress_early_logging(self):
        """Context manager to suppress logging during UI initialization"""
        # Store original levels
        root_logger = logging.getLogger()
        self._original_log_level = root_logger.level
        root_logger.setLevel(logging.CRITICAL)
        
        smartcash_logger = logging.getLogger('smartcash')
        self._original_smartcash_log_level = smartcash_logger.level
        smartcash_logger.setLevel(logging.CRITICAL)
        
        try:
            yield
        finally:
            # Restore original levels
            root_logger.setLevel(self._original_log_level)
            smartcash_logger.setLevel(self._original_smartcash_log_level)
    
    def _display_ui_component(self, ui_result: Union[Dict[str, Any], Any]) -> None:
        """
        Display UI component from initialization result.
        
        Args:
            ui_result: Result from _initialize_impl - either dict or widget
        """
        if isinstance(ui_result, dict):
            # Try to find the main UI component to display
            for key in ['ui', 'main_container', 'container']:
                if key in ui_result:
                    component = ui_result[key]
                    if self._is_displayable_widget(component):
                        display(component)
                        return
            
            # Look for any displayable widget
            for key, component in ui_result.items():
                if self._is_displayable_widget(component):
                    display(component)
                    return
                    
            self.logger.warning(f"⚠️ No displayable UI component found in {self.module_name} initialization")
                    
        elif self._is_displayable_widget(ui_result):
            # Direct widget
            display(ui_result)
        else:
            self.logger.warning(f"⚠️ Unexpected UI result type in {self.module_name}: {type(ui_result)}")
    
    def _is_displayable_widget(self, component) -> bool:
        """Check if a component is displayable."""
        if component is None:
            return False
            
        # Direct ipywidgets
        if hasattr(component, 'children') or hasattr(component, 'layout'):
            return True
        
        # Custom container classes with .container attribute
        if hasattr(component, 'container'):
            container = component.container
            if hasattr(container, 'children') or hasattr(container, 'layout'):
                display(container)
                return True
        
        # Classes with show() method
        if hasattr(component, 'show') and callable(component.show):
            display(component.show())
            return True
        
        return False
    
    def _display_error_component(self, error: Exception, error_msg: str = None) -> None:
        """
        Display beautiful error component using core error system.
        
        Args:
            error: The exception that occurred
            error_msg: Optional custom error message
        """
        # Use the error message or create one
        message = error_msg or f"Failed to initialize {self.module_name} UI"
        
        # Create beautiful error component
        error_component = create_error_component(
            error_message=f"{message}: {str(error)}",
            traceback=None,  # Don't show traceback by default in UI
            title=f"🚨 {self.module_name.title()} Initialization Error",
            error_type="error",
            show_traceback=False  # Can be toggled by user
        )
        
        # Display the error component
        if error_component and 'widget' in error_component:
            display(error_component['widget'])
        else:
            # Fallback to simple HTML display
            from IPython.display import HTML
            error_html = f"""
            <div style="color: #d32f2f; padding: 15px; border-left: 4px solid #d32f2f; 
                        margin: 10px 0; background: rgba(244, 67, 54, 0.05); border-radius: 4px;">
                <strong>🚨 {self.module_name.title()} Initialization Error</strong><br>
                <div style="margin-top: 8px; font-family: monospace; font-size: 13px;">
                    {message}: {str(error)}
                </div>
            </div>
            """
            display(HTML(error_html))
    
    def initialize_and_display(self, *args, **kwargs) -> None:
        """
        Initialize the module and display UI directly.
        
        This is the main entry point that should be called instead of initialize().
        It handles logging suppression, UI display, and error handling consistently.
        
        Args:
            *args: Arguments to pass to initialization
            **kwargs: Keyword arguments to pass to initialization
        """
        with self._suppress_early_logging():
            try:
                # Initialize the module components
                ui_result = self.initialize(*args, **kwargs)
                
                # Display the UI components
                self._display_ui_component(ui_result)
                
                # Log successful initialization (now that logging is restored)
                self.logger.info(f"✅ {self.module_name} UI displayed successfully")
                
            except Exception as e:
                # Display beautiful error component
                self._display_error_component(e)
                
                # Log the error (logging is restored in finally block)
                self.logger.error(f"❌ Failed to initialize {self.module_name} UI: {e}")
    
    @classmethod 
    def create_display_function(cls, 
                               module_name: str, 
                               parent_module: str = None,
                               initializer_class: type = None,
                               legacy_function: callable = None) -> callable:
        """
        Create a display function for a specific module.
        
        This factory method creates the initialize_*_ui functions used in cell files.
        
        Args:
            module_name: Name of the module (e.g., 'colab', 'dependency')
            parent_module: Parent module name (e.g., 'setup', 'dataset')
            initializer_class: Specific initializer class to use
            legacy_function: Existing function to wrap (fallback mode)
            
        Returns:
            Callable that initializes and displays the module UI
        """
        def initialize_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
            """Initialize and display the module UI."""
            
            import logging
            
            # Suppress early logging
            root_logger = logging.getLogger()
            original_level = root_logger.level
            root_logger.setLevel(logging.CRITICAL)
            
            smartcash_logger = logging.getLogger('smartcash')
            original_smartcash_level = smartcash_logger.level
            smartcash_logger.setLevel(logging.CRITICAL)
            
            try:
                # If legacy function provided, use it
                if legacy_function:
                    ui_result = legacy_function(config=config, **kwargs)
                else:
                    # Try to create initializer
                    if initializer_class:
                        try:
                            initializer = initializer_class(module_name, parent_module)
                            ui_result = initializer.initialize(config=config, **kwargs)
                        except Exception as e:
                            # Fallback to simple initialization
                            ui_result = None
                            raise e
                    else:
                        # Default behavior
                        initializer = cls(module_name, parent_module)
                        ui_result = initializer.initialize(config=config, **kwargs)
                
                # Restore logging now that UI should be ready
                root_logger.setLevel(original_level)
                smartcash_logger.setLevel(original_smartcash_level)
                
                # Display the UI components
                if ui_result:
                    cls._display_ui_component_static(ui_result)
                
            except Exception as e:
                # Restore logging in error case
                root_logger.setLevel(original_level)
                smartcash_logger.setLevel(original_smartcash_level)
                
                # Display beautiful error component
                cls._display_error_component_static(e, module_name)
        
        # Set function name for better debugging
        initialize_ui.__name__ = f"initialize_{module_name}_ui"
        initialize_ui.__doc__ = f"""
        Initialize and display the {module_name} UI.
        
        This function initializes the {module_name} module and displays its UI automatically.
        It manages logging to ensure no early prints appear before UI components are ready.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments
        """
        
        return initialize_ui
    
    @staticmethod
    def _display_ui_component_static(ui_result: Union[Dict[str, Any], Any]) -> None:
        """Static version of _display_ui_component for use in factory function"""
        if isinstance(ui_result, dict):
            # Try to find the main UI component to display
            for key in ['ui', 'main_container', 'container']:
                if key in ui_result:
                    component = ui_result[key]
                    if DisplayInitializer._is_displayable_widget_static(component):
                        return
            
            # Look for any displayable widget
            for key, component in ui_result.items():
                if DisplayInitializer._is_displayable_widget_static(component):
                    return
                        
        elif DisplayInitializer._is_displayable_widget_static(ui_result):
            # Direct widget
            pass
    
    @staticmethod
    def _is_displayable_widget_static(component) -> bool:
        """Static version of _is_displayable_widget."""
        if component is None:
            return False
            
        # Direct ipywidgets
        if hasattr(component, 'children') or hasattr(component, 'layout'):
            display(component)
            return True
        
        # Custom container classes with .container attribute
        if hasattr(component, 'container'):
            container = component.container
            if hasattr(container, 'children') or hasattr(container, 'layout'):
                display(container)
                return True
        
        # Classes with show() method
        if hasattr(component, 'show') and callable(component.show):
            display(component.show())
            return True
        
        return False
    
    @staticmethod
    def _display_error_component_static(error: Exception, module_name: str) -> None:
        """Static version of _display_error_component for use in factory function"""
        message = f"Failed to initialize {module_name} UI"
        
        # Try to create beautiful error component
        try:
            error_component = create_error_component(
                error_message=f"{message}: {str(error)}",
                title=f"🚨 {module_name.title()} Initialization Error",
                error_type="error",
                show_traceback=False
            )
            
            if error_component and 'widget' in error_component:
                display(error_component['widget'])
                return
        except:
            pass
        
        # Fallback to simple HTML display
        from IPython.display import HTML
        error_html = f"""
        <div style="color: #d32f2f; padding: 15px; border-left: 4px solid #d32f2f; 
                    margin: 10px 0; background: rgba(244, 67, 54, 0.05); border-radius: 4px;">
            <strong>🚨 {module_name.title()} Initialization Error</strong><br>
            <div style="margin-top: 8px; font-family: monospace; font-size: 13px;">
                {message}: {str(error)}
            </div>
        </div>
        """
        display(HTML(error_html))


# Convenience function for backward compatibility and easy import
def create_ui_display_function(module_name: str, 
                              parent_module: str = None,
                              initializer_class: type = None,
                              legacy_function: callable = None) -> callable:
    """
    Create a UI display function for any module.
    
    This is a convenience function that modules can use to create their
    initialize_*_ui functions with consistent behavior.
    
    Example:
        # In colab_initializer.py
        initialize_colab_ui = create_ui_display_function(
            'colab', 'setup', ColabInitializer
        )
    
    Args:
        module_name: Name of the module
        parent_module: Parent module name  
        initializer_class: Specific initializer class to use
        
    Returns:
        Function that initializes and displays the module UI
    """
    return DisplayInitializer.create_display_function(
        module_name, parent_module, initializer_class, legacy_function
    )
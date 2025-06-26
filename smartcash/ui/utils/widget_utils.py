"""
Utility functions for working with IPython widgets.
"""
from typing import Any, Dict, Optional, Union
from IPython.display import display

def display_widget(widget: Any) -> None:
    """
    Display a widget, handling both direct widget and dictionary with common container keys.
    
    Args:
        widget: The widget to display. Can be:
            - A widget object directly
            - A dict with one of these keys containing a widget:
              'ui', 'container', 'main_container'
    """
    if widget is None:
        return
        
    if isinstance(widget, dict):
        # Try common widget container keys in order of preference
        for key in ['ui', 'container', 'main_container']:
            if key in widget and widget[key] is not None:
                display(widget[key])
                return
        
        # If no container keys found but dict is not empty, try to display first widget found
        for value in widget.values():
            if hasattr(value, '_ipython_display_'):  # Check if it's a widget
                display(value)
                return
    
    # Default case: display the widget directly
    display(widget)

def safe_display(
    widget: Any, 
    condition: bool = True
) -> Optional[Any]:
    """
    Safely display a widget if the condition is True.
    
    Args:
        widget: The widget to display. Can be a widget or a dict with a 'ui' key.
        condition: If True, display the widget. Defaults to True.
        
    Returns:
        The original widget for method chaining.
    """
    if condition and widget is not None:
        display_widget(widget)
    return widget

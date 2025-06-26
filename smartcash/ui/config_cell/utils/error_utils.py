"""
File: smartcash/ui/config_cell/utils/error_utils.py
Deskripsi: Error handling utilities for config cell
"""
from typing import Dict, Any, Optional
from IPython.display import display, HTML

from smartcash.ui.components.error.error_component import create_error_component
def create_error_fallback(
    error_message: str, 
    traceback: Optional[str] = None,
    **kwargs
) -> Any:
    """Create a fallback UI for error states using the standard ErrorComponent
    
    Args:
        error_message: Error message to display
        traceback: Optional traceback information
        **kwargs: Additional arguments passed to create_error_component
        
    Returns:
        A widget containing the error UI
    """
    error_component = create_error_component(
        error_message=error_message,
        traceback=traceback,
        error_type="error",
        show_traceback=bool(traceback),
        **kwargs
    )
    
    # Display the error
    display(error_component['error_widget'])
    
    # Get the widget from the component
    widget = error_component.get('widget')
    
    # If we have a widget, return it directly
    if widget is not None:
        return widget
        
    # Fallback to container if widget is not available
    if 'container' in error_component:
        return error_component['container']
        
    # Last resort: create a simple error widget
    title = kwargs.get('title', 'Error')
    return HTML(f'<div style="color:red; padding: 10px; border: 1px solid red;">{title}: {error_message}</div>')

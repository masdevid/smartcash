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
    
    # Try to get widget in order of preference: widget > container > error_widget
    widget = next(
        (w for w in [
            error_component.get('widget'),
            error_component.get('container'),
            error_component.get('error_widget')
        ] if w is not None),
        None
    )
    
    return (widget or HTML(f'<div style="color:red;padding:10px;border:1px solid red;margin:5px 0;border-radius:4px;background-color:#fff0f0"><strong>{kwargs.get("title", "Error")}:</strong> {error_message}</div>'))

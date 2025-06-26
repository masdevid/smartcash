"""
File: smartcash/ui/config_cell/utils/error_utils.py
Deskripsi: Error handling utilities for config cell
"""
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.components.error.error_component import create_error_component

def create_error_fallback(
    error_message: str, 
    traceback: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a fallback UI for error states using the standard ErrorComponent
    
    Args:
        error_message: Error message to display
        traceback: Optional traceback information
        **kwargs: Additional arguments passed to create_error_component
        
    Returns:
        Dictionary containing error UI components with keys:
        - container: The main error widget
        - error: The error message
        - initialized: Always False for error states
    """
    error_component = create_error_component(
        error_message=error_message,
        traceback=traceback,
        error_type="error",
        show_traceback=bool(traceback),
        **kwargs
    )
    
    # Ensure the error is displayed
    display(error_component['widget'])
    
    return {
        'container': error_component['widget'],
        'error': error_message,
        'initialized': False
    }

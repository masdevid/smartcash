"""
File: smartcash/ui/config_cell/utils/error_utils.py
Deskripsi: Error handling utilities for config cell
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_error_fallback(error_message: str, traceback: Optional[str] = None) -> Dict[str, Any]:
    """Create a fallback UI for error states
    
    Args:
        error_message: Error message to display
        traceback: Optional traceback information
        
    Returns:
        Dictionary containing error UI components
    """
    # Create error message with optional traceback
    error_content = [
        widgets.HTML(
            f'<div style="color: #f44336; font-weight: bold; margin-bottom: 10px;">'
            f'⚠️ {error_message}'
            '</div>'
        )
    ]
    
    if traceback:
        error_content.append(
            widgets.Textarea(
                value=traceback,
                layout={"height": "100px", "width": "100%"},
                disabled=True
            )
        )
    
    # Create container with error styling
    container = widgets.VBox(
        error_content,
        layout={
            "border": "1px solid #f44336",
            "border_radius": "5px",
            "padding": "10px",
            "margin": "5px 0"
        }
    )
    
    return {
        'container': container,
        'error': error_message,
        'initialized': False
    }

"""Widget utilities for UI components."""

import gc
import ipywidgets as widgets
from typing import List, Any


def hide_stray_accordions(safe_accordions: List[Any] = None) -> int:
    """
    Hide any stray accordion widgets that might be causing UI issues.
    
    This is a utility function to clean up any orphaned accordion widgets
    that might be causing duplicate UI elements to appear.
    
    Args:
        safe_accordions: List of accordion widgets that should NOT be hidden
        
    Returns:
        Number of accordions that were hidden
    """
    if safe_accordions is None:
        safe_accordions = []
        
    hidden_count = 0
    
    for obj in gc.get_objects():
        if isinstance(obj, widgets.Accordion) and obj not in safe_accordions:
            if hasattr(obj, 'layout'):
                obj.layout.display = 'none'
                hidden_count += 1
                
    return hidden_count

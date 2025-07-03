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
    
    # Get all objects and filter for Accordion widgets
    for obj in gc.get_objects():
        try:
            # Check if it's an Accordion widget without using 'in' operator
            # This avoids the FutureWarning from torch.distributed
            is_accordion = False
            if hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
                if obj.__class__.__name__ == 'Accordion':
                    is_accordion = True
                    
            if not is_accordion:
                continue
                    
            # Check if it's in our safe list
            is_safe = False
            for safe_obj in safe_accordions:
                if obj is safe_obj:  # Use identity comparison instead of equality
                    is_safe = True
                    break
            
            # Check if it's an orphaned accordion (not attached to any parent)
            is_orphaned = True
            if hasattr(obj, '_parent') and obj._parent is not None:
                is_orphaned = False
            
            # Hide if it's an accordion and either orphaned or not in our safe list
            if not is_safe or is_orphaned:
                if hasattr(obj, 'layout'):
                    obj.layout.display = 'none'
                    hidden_count += 1
        except Exception:
            # Skip any objects that cause errors during inspection
            continue
                
    return hidden_count

"""
File: smartcash/ui/components/dialog/__init__.py
Deskripsi: Reusable dialog components untuk confirmation dan user interaction
"""

# New simplified dialog components (recommended)
from .simple_dialog import (
    SimpleDialog,
    create_simple_dialog,
    show_confirmation_dialog as simple_show_confirmation_dialog,
    show_info_dialog as simple_show_info_dialog,
    show_success_dialog,
    show_warning_dialog,
    show_error_dialog
)

# Legacy components (maintained for backward compatibility)
from .confirmation_dialog import (
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible,
    create_confirmation_area
)

__all__ = [
    # New simplified dialog components
    'SimpleDialog',
    'create_simple_dialog',
    'simple_show_confirmation_dialog',
    'simple_show_info_dialog',
    'show_success_dialog',
    'show_warning_dialog',
    'show_error_dialog',
    
    # Legacy components (backward compatibility)
    'show_confirmation_dialog',
    'show_info_dialog', 
    'clear_dialog_area',
    'is_dialog_visible',
    'create_confirmation_area'
]
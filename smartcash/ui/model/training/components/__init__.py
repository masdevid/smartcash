"""
File: smartcash/ui/model/training/components/__init__.py
Training UI components package - Updated for unified training pipeline.
"""

# Unified training components (new)
from .unified_training_ui import (
    create_unified_training_ui,
    update_training_buttons_state,
    update_summary_display
)

from .unified_training_form import (
    create_unified_training_form
)

# Legacy components removed - use unified system instead

__all__ = [
    # Unified training components (primary)
    'create_unified_training_ui',
    'update_training_buttons_state', 
    'update_summary_display',
    'create_unified_training_form',
    
]
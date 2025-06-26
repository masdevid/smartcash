"""
File: smartcash/ui/dataset/split/__init__.py
Deskripsi: Ekspor utilitas dan fungsi split dataset
"""

from .split_init import (
    create_split_config_cell,
    SplitConfigHandler
)

# For backward compatibility
create_split_config_ui = create_split_config_cell

# Public API
def initialize_split_ui(config=None):
    """Initialize the split dataset UI with optional configuration.
    
    This is the main entry point for the split dataset functionality.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of UI components
    """
    return create_split_config_cell(config)

create_split_init = create_split_config_cell

__all__ = [
    'initialize_split_ui',
    'create_split_init',
    'create_split_config_ui',  # Backward compatibility
    'create_split_config_cell',
    'SplitConfigHandler'
]

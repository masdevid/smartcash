"""
File: smartcash/ui/dataset/visualization/configs/visualization_defaults.py
Description: Default configuration values for the visualization module
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class VisualizationDefaults:
    """Default configuration values for the visualization module."""
    
    # List of splits to display
    SPLITS: List[str] = field(default_factory=lambda: ['train', 'valid', 'test'])
    
    # Percentage display format
    PERCENTAGE_FORMAT: str = '{:.1f}%'
    
    # Colors for each split
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'train': '#4CAF50',  # Green
        'valid': '#2196F3',  # Blue
        'test': '#FF9800'    # Orange
    })
    
    # Refresh interval in seconds (0 to disable auto-refresh)
    REFRESH_INTERVAL: int = 0
    
    # Show log accordion by default
    SHOW_LOG_ACCORDION: bool = True
    
    # Show refresh button
    SHOW_REFRESH_BUTTON: bool = True
    
    # Default dataset path
    DEFAULT_DATASET_PATH: Optional[str] = None

# Default configuration instance
DEFAULT_CONFIG = VisualizationDefaults()

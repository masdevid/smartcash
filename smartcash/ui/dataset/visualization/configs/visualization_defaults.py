"""
File: smartcash/ui/dataset/visualization/configs/visualization_defaults.py
Deskripsi: Default configuration untuk visualization module
"""

from typing import Dict, List, Any, Optional

def get_default_visualization_config() -> Dict[str, Any]:
    """Get default visualization configuration"""
    return {
        'module_name': 'visualization',
        'version': '1.0.0',
        'created_by': 'SmartCash',
        'description': 'Visualisasi dataset dengan berbagai opsi tampilan interaktif',
        
        # Data splits configuration
        'splits': ['train', 'valid', 'test'],
        'split_colors': {
            'train': '#4CAF50',  # Green
            'valid': '#2196F3',  # Blue
            'test': '#FF9800'    # Orange
        },
        
        # Display settings
        'display': {
            'percentage_format': '{:.1f}%',
            'refresh_interval': 0,  # in seconds, 0 to disable
            'show_log_accordion': True,
            'show_refresh_button': True,
            'default_theme': 'light',
            'chart_theme': 'plotly_white'
        },
        
        # Path settings
        'paths': {
            'default_dataset': None,
            'export_dir': './exports/visualization',
            'cache_dir': './.cache/visualization'
        },
        
        # Data directory configuration
        'data_dir': 'data',  # Default data directory
        
        # UI components visibility
        'ui': {
            'show_statistics': True,
            'show_distributions': True,
            'show_preview': True,
            'show_controls': True
        },
        
        # Performance settings
        'performance': {
            'max_data_points': 10000,
            'sampling_enabled': True,
            'use_webgl': True,
            'cache_enabled': True
        }
    }

def get_visualization_colors() -> Dict[str, str]:
    """Get default color scheme for visualizations"""
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ffbb78',
        'info': '#98df8a',
        'light': '#f7f7f7',
        'dark': '#2c3e50',
        'background': '#ffffff',
        'text': '#2c3e50'
    }

# Default configuration instance
DEFAULT_CONFIG = get_default_visualization_config()

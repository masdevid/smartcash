"""
Log level definitions and utilities for the LogAccordion component.
"""

from enum import Enum
from typing import Dict, Any


class LogLevel(Enum):
    """Log level enumeration with associated styles and icons."""
    DEBUG = 'debug'
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


# Define log level styles with better colors, backgrounds, and text colors for compact format
LOG_LEVEL_STYLES: Dict[LogLevel, Dict[str, Any]] = {
    LogLevel.DEBUG: {
        'color': '#6c757d', 
        'bg': 'rgba(108, 117, 125, 0.08)', 
        'text_color': '#495057',
        'icon': '🔹', 
        'border': '1px solid #dee2e6'
    },
    LogLevel.INFO: {
        'color': '#0d6efd', 
        'bg': 'rgba(13, 110, 253, 0.08)', 
        'text_color': '#0d47a1',
        'icon': 'ℹ️', 
        'border': '1px solid #b3d1ff'
    },
    LogLevel.SUCCESS: {
        'color': '#198754', 
        'bg': 'rgba(25, 135, 84, 0.08)', 
        'text_color': '#0d5130',
        'icon': '✅', 
        'border': '1px solid #a3cfbb'
    },
    LogLevel.WARNING: {
        'color': '#ff9800', 
        'bg': 'rgba(255, 152, 0, 0.08)', 
        'text_color': '#e65100',
        'icon': '⚠️', 
        'border': '1px solid #ffe0b2'
    },
    LogLevel.ERROR: {
        'color': '#d32f2f', 
        'bg': 'rgba(211, 47, 47, 0.08)', 
        'text_color': '#b71c1c',
        'icon': '❌', 
        'border': '1px solid #ffcdd2'
    },
    LogLevel.CRITICAL: {
        'color': '#b71c1c', 
        'bg': 'rgba(183, 28, 28, 0.12)', 
        'text_color': '#7f0000',
        'icon': '🔥', 
        'border': '1px solid #ffcdd2'
    }
}


def get_log_level_style(level: LogLevel) -> Dict[str, Any]:
    """Get the style dictionary for a log level.
    
    Args:
        level: The log level to get style for
        
    Returns:
        Dictionary containing color, background, and icon for the level
    """
    return LOG_LEVEL_STYLES.get(level, LOG_LEVEL_STYLES[LogLevel.INFO])

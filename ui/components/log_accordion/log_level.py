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


# Define log level styles
LOG_LEVEL_STYLES: Dict[LogLevel, Dict[str, Any]] = {
    LogLevel.DEBUG: {'color': '#6c757d', 'bg': '#f8f9fa', 'icon': '🔍'},
    LogLevel.INFO: {'color': '#0d6efd', 'bg': '#e7f1ff', 'icon': 'ℹ️'},
    LogLevel.SUCCESS: {'color': '#198754', 'bg': '#e7f8f0', 'icon': '✅'},
    LogLevel.WARNING: {'color': '#ffc107', 'bg': '#fff8e6', 'icon': '⚠️'},
    LogLevel.ERROR: {'color': '#dc3545', 'bg': '#fdf0f2', 'icon': '❌'},
    LogLevel.CRITICAL: {'color': '#dc3545', 'bg': '#fdf0f2', 'icon': '🔥'}
}


def get_log_level_style(level: LogLevel) -> Dict[str, Any]:
    """Get the style dictionary for a log level.
    
    Args:
        level: The log level to get style for
        
    Returns:
        Dictionary containing color, background, and icon for the level
    """
    return LOG_LEVEL_STYLES.get(level, LOG_LEVEL_STYLES[LogLevel.INFO])

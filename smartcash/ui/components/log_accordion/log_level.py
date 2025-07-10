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


# Define log level styles with better colors and icons
LOG_LEVEL_STYLES: Dict[LogLevel, Dict[str, Any]] = {
    LogLevel.DEBUG: {'color': '#6c757d', 'bg': '#f8f9fa', 'icon': '🔹', 'border': '1px solid #dee2e6'},
    LogLevel.INFO: {'color': '#0d6efd', 'bg': '#e7f1ff', 'icon': 'ℹ️', 'border': '1px solid #b3d1ff'},
    LogLevel.SUCCESS: {'color': '#198754', 'bg': '#e7f8f0', 'icon': '✓', 'border': '1px solid #a3cfbb'},
    LogLevel.WARNING: {'color': '#ff9800', 'bg': '#fff3e0', 'icon': '⚠️', 'border': '1px solid #ffe0b2'},
    LogLevel.ERROR: {'color': '#d32f2f', 'bg': '#ffebee', 'icon': '✗', 'border': '1px solid #ffcdd2'},
    LogLevel.CRITICAL: {'color': '#b71c1c', 'bg': '#ffebee', 'icon': '🔥', 'border': '1px solid #ffcdd2'}
}


def get_log_level_style(level: LogLevel) -> Dict[str, Any]:
    """Get the style dictionary for a log level.
    
    Args:
        level: The log level to get style for
        
    Returns:
        Dictionary containing color, background, and icon for the level
    """
    return LOG_LEVEL_STYLES.get(level, LOG_LEVEL_STYLES[LogLevel.INFO])

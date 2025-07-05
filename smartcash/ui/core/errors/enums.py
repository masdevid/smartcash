"""
Error level enumerations and constants.

This module defines the error levels used throughout the SmartCash UI Core
for consistent error handling and logging.
"""
from enum import Enum, auto
from typing import Dict, Any


class ErrorLevel(Enum):
    """
    Defines the severity levels for error handling and logging.
    
    Attributes:
        DEBUG: Detailed information, typically of interest only when diagnosing problems.
        INFO: Confirmation that things are working as expected.
        WARNING: An indication that something unexpected happened.
        ERROR: Due to a more serious problem, the software has not been able to perform some function.
        CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    """
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    
    @property
    def color(self) -> str:
        """
        Get the color associated with this error level for UI display.
        
        Returns:
            str: A CSS color code for the error level.
        """
        return {
            ErrorLevel.DEBUG: "#888888",    # Gray
            ErrorLevel.INFO: "#2196F3",     # Blue
            ErrorLevel.WARNING: "#FF9800",  # Orange
            ErrorLevel.ERROR: "#F44336",    # Red
            ErrorLevel.CRITICAL: "#9C27B0"  # Purple
        }[self]
    
    @property
    def icon(self) -> str:
        """
        Get the icon associated with this error level.
        
        Returns:
            str: An emoji icon representing the error level.
        """
        return {
            ErrorLevel.DEBUG: "🐛",
            ErrorLevel.INFO: "ℹ️",
            ErrorLevel.WARNING: "⚠️",
            ErrorLevel.ERROR: "❌",
            ErrorLevel.CRITICAL: "🚨"
        }[self]
    
    def __str__(self) -> str:
        """
        Get the string representation of the error level.
        
        Returns:
            str: The name of the error level in lowercase.
        """
        return self.name.lower()

"""
Configuration management for the dataset split module.

This module contains configuration-related constants and utilities.
"""

# Import only the configuration constants to avoid circular imports
from .split_defaults import DEFAULT_SPLIT_CONFIG, VALIDATION_RULES

__all__ = [
    'DEFAULT_SPLIT_CONFIG',
    'VALIDATION_RULES'
]

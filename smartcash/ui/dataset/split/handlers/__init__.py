"""
Dataset Split Handlers Module.

This module contains handler classes for the dataset split functionality,
including configuration management and UI event handling.
"""

# Import the handler class directly to avoid circular imports
class SplitConfigHandler:
    """Lazy loader for SplitConfigHandler to avoid circular imports."""
    def __new__(cls, *args, **kwargs):
        from .config_handler import SplitConfigHandler as _SplitConfigHandler
        return _SplitConfigHandler(*args, **kwargs)

__all__ = [
    'SplitConfigHandler'
]

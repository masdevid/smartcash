"""
File: smartcash/ui/types.py
Deskripsi: Shared type definitions for the UI components
"""
from enum import Enum, auto
from typing import Callable, Optional, TypeVar

# Type aliases for callback functions
StatusCallback = Callable[[str], None]  # Function that takes a status message and returns None

class ProgressTrackerType(Enum):
    """Enum for different types of progress trackers"""
    NONE = auto()
    SINGLE = auto()
    DUAL = auto()

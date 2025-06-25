"""
File: smartcash/ui/setup/dependency/utils/system/__init__.py

System information and compatibility utilities.

This module provides functions for gathering system information and checking
system compatibility with SmartCash requirements.
"""

from .info import *

# Re-export all symbols from submodules
__all__ = info.__all__ if hasattr(info, '__all__') else []

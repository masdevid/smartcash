"""
Setup Module - Environment and dependency setup for SmartCash

This module provides interfaces for setting up the SmartCash environment,
including dependency management and Colab environment configuration.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/setup/__init__.py
"""

from . import colab
from . import dependency

# Export main classes and functions
__all__ = [
    # Submodules
    'colab',
    'dependency'
]

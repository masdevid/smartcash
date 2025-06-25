"""
File: smartcash/ui/setup/dependency/utils/reporting/__init__.py

Reporting utilities for dependency management.

This module provides functionality to generate various reports and summaries
about the system, package status, and installation results.
"""

from .generators import *

# Re-export all symbols from submodules
__all__ = generators.__all__ if hasattr(generators, '__all__') else []

"""
Core validation modules for UI components.

This package provides validation utilities for ensuring consistency
and correctness across UI modules.
"""

from .button_validator import ButtonHandlerValidator, ButtonValidationResult

__all__ = [
    'ButtonHandlerValidator',
    'ButtonValidationResult'
]
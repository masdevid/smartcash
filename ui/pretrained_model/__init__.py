"""
File: smartcash/ui/pretrained_model/__init__.py
Deskripsi: Package initialization untuk pretrained model
"""

from .pretrained_init import initialize_pretrained_model_ui
from . import components  # Import relatif aman

__all__ = ['initialize_pretrained_model_ui']

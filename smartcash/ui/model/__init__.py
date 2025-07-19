"""
Model Module - Comprehensive model management for SmartCash

This module provides interfaces for model management including training, evaluation,
and backbone model integration.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/__init__.py
"""
from . import pretrained
from . import backbone
from . import training
from . import evaluation

# Export main classes and functions
__all__ = [
    # Core components
    'pretrained',
    'backbone',
    'training',
    'evaluation',
]
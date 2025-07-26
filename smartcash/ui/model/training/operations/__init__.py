"""
File: smartcash/ui/model/training/operations/__init__.py
Training operations package - Updated for unified training pipeline.
"""

# Unified training operation (primary)
from .unified_training_operation import UnifiedTrainingOperation

# Legacy operations removed - use unified system instead

__all__ = [
    # Unified training operation (primary)
    'UnifiedTrainingOperation',
    
]
"""
Core mixins for UI modules to reduce code duplication.

This module provides common functionality mixins that can be used across
UI modules to provide consistent behavior and reduce code duplication.
"""

from .configuration_mixin import ConfigurationMixin
from .operation_mixin import OperationMixin
from .logging_mixin import LoggingMixin
# ProgressTrackingMixin removed - use operation_container.update_progress() instead
from .button_handler_mixin import ButtonHandlerMixin
from .validation_mixin import ValidationMixin
from .display_mixin import DisplayMixin
from .colab_secrets_mixin import ColabSecretsMixin

__all__ = [
    'ConfigurationMixin',
    'OperationMixin', 
    'LoggingMixin',
    # 'ProgressTrackingMixin',  # Removed - use operation_container instead
    'ButtonHandlerMixin',
    'ValidationMixin',
    'DisplayMixin',
    'ColabSecretsMixin'
]
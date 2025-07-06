"""
File: smartcash/ui/initializers/__init__.py

⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.
Please update your imports to use smartcash.ui.core.initializers instead.
"""
import warnings

# Issue deprecation warning
warnings.warn(
    "The 'smartcash.ui.initializers' module is deprecated and will be removed in a future version. "
    "Please update your imports to use 'smartcash.ui.core.initializers' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Forward imports from the new location
from smartcash.ui.core.initializers import (
    CommonInitializer,
    BaseInitializer,
    ConfigurableInitializer,
    ModuleInitializer,
    ColabEnvInitializer
)

__all__ = [
    'CommonInitializer',
    'BaseInitializer',
    'ConfigurableInitializer',
    'ModuleInitializer',
    'ColabEnvInitializer'
]

"""Colab setup UI components package.

This module provides UI components for the Colab environment setup.
"""

from importlib import import_module
from types import ModuleType
from typing import Any, Dict, TYPE_CHECKING

from .colab_ui import create_colab_ui_components

# For backward compatibility
__all__ = ["create_colab_ui_components"]

# Lazy import for backward compatibility
def _lazy_import(name: str) -> ModuleType:  # pragma: no cover
    return import_module(name)


def __getattr__(name: str):  # pragma: no cover
    # For backward compatibility
    if name == "create_env_config_ui" or name == "create_colab_ui":
        import warnings
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use create_colab_ui_components instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return _lazy_import("smartcash.ui.setup.colab.components.colab_ui").create_colab_ui_components
    
    if name == "create_colab_ui_components":
        return _lazy_import("smartcash.ui.setup.colab.components.colab_ui").create_colab_ui_components
        
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["create_colab_ui_components"]

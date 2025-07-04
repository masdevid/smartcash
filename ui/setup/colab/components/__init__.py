"""Colab setup UI components package."""

from importlib import import_module
from types import ModuleType
from typing import Any, Dict


def _lazy_import(name: str) -> ModuleType:  # pragma: no cover
    return import_module(name)


def __getattr__(name: str):  # pragma: no cover
    # For backward compatibility
    if name == "create_env_config_ui":
        import warnings
        warnings.warn(
            "create_env_config_ui is deprecated and will be removed in a future version. "
            "Use create_colab_ui instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return _lazy_import("smartcash.ui.setup.colab.components.ui_components").create_colab_ui
    
    if name == "create_colab_ui":
        return _lazy_import("smartcash.ui.setup.colab.components.ui_components").create_colab_ui
        
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["create_colab_ui"]

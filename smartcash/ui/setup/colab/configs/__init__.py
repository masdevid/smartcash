"""Package untuk konfigurasi default Colab setup."""

from importlib import import_module
from types import ModuleType

__all__ = ["DEFAULT_CONFIG"]


def _lazy_import(name: str) -> ModuleType:  # pragma: no cover
    return import_module(name)


# Expose DEFAULT_CONFIG lazily


def __getattr__(name):  # pragma: no cover
    if name == "DEFAULT_CONFIG":
        return _lazy_import("smartcash.ui.setup.colab.configs.defaults").DEFAULT_CONFIG
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

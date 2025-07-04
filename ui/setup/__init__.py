"""
File: smartcash/ui/setup/__init__.py
Deskripsi: Entry point untuk submodul `smartcash.ui.setup`.

Refaktor:
    - Mengganti import eager menjadi lazy import untuk menghindari import berat
      ketika hanya membutuhkan sebagian kecil fungsionalitas.
    - Menyediakan fallback backward-compatibility melalui `__getattr__`.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

# ---------------------------------------------------------------------------
# Lazy Import Utilities
# ---------------------------------------------------------------------------

def _lazy_import(module_path: str) -> ModuleType:
    """Import modul secara lazy dan kembalikan reference-nya."""
    return importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Public Lazy Attributes
# ---------------------------------------------------------------------------

_lazy_attrs = {
    "initialize_env_config_ui": "smartcash.ui.setup.env_config.env_config_initializer.initialize_env_config_ui",
    "initialize_dependency_ui": "smartcash.ui.setup.dependency.dependency_initializer.initialize_dependency_ui",
    "colab": "smartcash.ui.setup.colab",
    "initialize_colab_ui": "smartcash.ui.setup.colab.initializer.initialize_colab_env_ui",
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _lazy_attrs:
        module_path, func_name = _lazy_attrs[name].rsplit(".", 1)
        module = _lazy_import(module_path)
        return getattr(module, func_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__: list[str] = list(_lazy_attrs.keys())

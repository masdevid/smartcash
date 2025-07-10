"""
file_path: smartcash/ui/setup/colab/__init__.py
Deskripsi: Modul utama untuk inisialisasi dan manajemen UI di lingkungan Google Colab.

Note: This module requires explicit initialization before use.
"""

# Lazy imports to prevent automatic initialization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .colab_initializer import (
        initialize_colab_ui,
        display_colab_ui,
        get_colab_components,
        get_colab_initializer,
        get_colab_env_initializer,
        ColabInitializer,
        ColabDisplayInitializer
    )

# Import the main functions directly to make them available at package level
from .colab_initializer import (
    initialize_colab_ui,
    display_colab_ui,
    get_colab_components,
    get_colab_initializer,
    get_colab_env_initializer,
    ColabInitializer,
    ColabDisplayInitializer
)

__all__ = [
    "initialize_colab_ui",
    "display_colab_ui",
    "get_colab_components",
    "get_colab_initializer",
    "get_colab_env_initializer",
    "ColabInitializer",
    "ColabDisplayInitializer"
]

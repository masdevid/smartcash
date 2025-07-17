"""
file_path: smartcash/ui/setup/colab/__init__.py
Deskripsi: Modul utama untuk inisialisasi dan manajemen UI di lingkungan Google Colab.

Note: This module has been refactored to use UIModule pattern.
"""

# Import new UIModule functions (preferred approach)
from .colab_uimodule import (
    ColabUIModule,
    create_colab_uimodule,
    get_colab_uimodule,
    reset_colab_uimodule,
    initialize_colab_ui,
    get_colab_components
)

__all__ = [
    # UIModule pattern (current implementation)
    "ColabUIModule",
    "create_colab_uimodule", 
    "get_colab_uimodule",
    "reset_colab_uimodule",
    "initialize_colab_ui",
    "get_colab_components"
]

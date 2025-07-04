"""Unit test untuk memastikan modul constants pada smartcash.ui.setup.colab
bisa di-impor tanpa error dan objek kunci tersedia.
"""

import importlib
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("module_path, attr_name", [
    ("smartcash.ui.setup.colab.constants", "REQUIRED_FOLDERS"),
    ("smartcash.ui.setup.colab.constants", "SOURCE_DIRECTORIES"),
    ("smartcash.ui.setup.colab.constants", "SYMLINK_MAP"),
    ("smartcash.ui.setup.colab.constants", "STATUS_MESSAGES"),
])
def test_import_and_attr_exists(module_path: str, attr_name: str) -> None:
    """Pastikan modul dapat diimport dan atribut tersedia."""

    # Hilangkan mock MagicMock pada `smartcash.ui.setup` jika ada, agar import berjalan normal
    if (
        'smartcash.ui.setup' in sys.modules
        and not isinstance(sys.modules['smartcash.ui.setup'], ModuleType)
    ):
        sys.modules.pop('smartcash.ui.setup')

    module = importlib.import_module(module_path)
    assert hasattr(module, attr_name), f"{attr_name} tidak ditemukan pada {module_path}"

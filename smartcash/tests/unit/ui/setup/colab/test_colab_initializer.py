"""Unit test untuk memastikan initializer Colab dapat diimpor.

Fokus test: hanya verifikasi import sukses dan attribute utama tersedia, tanpa
mengeksekusi proses `initialize` penuh agar tidak memicu dependensi berat.
"""

import importlib
import sys
from types import ModuleType

import pytest

MODULE_PATH = "smartcash.ui.setup.colab.colab_initializer"

@pytest.mark.parametrize("attr_name", [
    "ColabEnvInitializer",
    "initialize_colab_env_ui",
])
def test_initializer_import(attr_name: str) -> None:
    """Pastikan modul initializer dapat diimpor dan atribut tersedia."""

    # Bersihkan mock MagicMock pada `smartcash.ui.setup` jika ada
    if (
        "smartcash.ui.setup" in sys.modules
        and not isinstance(sys.modules["smartcash.ui.setup"], ModuleType)
    ):
        sys.modules.pop("smartcash.ui.setup")

    module = importlib.import_module(MODULE_PATH)
    assert hasattr(module, attr_name), f"{attr_name} tidak ditemukan pada {MODULE_PATH}"

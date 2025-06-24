"""
Konfigurasi pytest untuk testing SmartCash
"""
import os
import sys
import pytest
from pathlib import Path

# Tambahkan path root project ke PYTHONPATH
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

# Konfigurasi pytest
def pytest_configure(config):
    """Konfigurasi pytest"""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")

# Fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Direktori untuk data test"""
    return os.path.join(ROOT_DIR, "tests", "data")

@pytest.fixture(scope="session")
def sample_image_path(test_data_dir):
    """Path ke sample image untuk testing"""
    return os.path.join(test_data_dir, "sample_currency.jpg")

@pytest.fixture(scope="module")
def pretrained_model():
    """Fixture untuk model pretrained"""
    # Di sini bisa diisi dengan inisialisasi model
    return {"status": "model_loaded"}

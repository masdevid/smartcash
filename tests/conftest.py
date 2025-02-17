# File: tests/conftest.py
# Author: Alfrida Sabar
# Deskripsi: Konfigurasi global untuk pengujian pytest

import os
import sys
import pytest

# Tambahkan root project ke path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def pytest_configure(config):
    """Konfigurasi global untuk pytest"""
    # Tambahkan marker kustom
    config.addinivalue_line(
        "markers", 
        "slow: penanda untuk tes yang membutuhkan waktu lama"
    )
    config.addinivalue_line(
        "markers", 
        "integration: penanda untuk tes integrasi"
    )

def pytest_addoption(parser):
    """Tambahkan opsi kustom untuk pytest"""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="jalankan tes yang ditandai sebagai lambat"
    )

def pytest_collection_modifyitems(config, items):
    """Modifikasi koleksi tes berdasarkan marker"""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="butuh flag --runslow untuk dijalankan")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
"""
File: smartcash/ui/dataset/split/tests/conftest.py
Deskripsi: Konfigurasi untuk pytest
"""

import pytest
import os
import sys

# Pastikan path project ada di PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Fixture untuk setup environment test
@pytest.fixture
def test_env():
    """Fixture untuk menyediakan environment test."""
    from unittest.mock import MagicMock
    env = MagicMock()
    env.base_dir = '/dummy/path'
    return env

# Fixture untuk setup config test
@pytest.fixture
def test_config():
    """Fixture untuk menyediakan config test."""
    return {
        'split': {
            'enabled': True,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_seed': 42,
            'stratify': True
        }
    }

# Fixture untuk setup UI components
@pytest.fixture
def ui_components():
    """Fixture untuk menyediakan UI components test."""
    from unittest.mock import MagicMock
    return {
        'train_slider': MagicMock(value=0.7),
        'val_slider': MagicMock(value=0.15),
        'test_slider': MagicMock(value=0.15),
        'random_seed': MagicMock(value=42),
        'stratified_checkbox': MagicMock(value=True),
        'enabled_checkbox': MagicMock(value=True),
        'save_button': MagicMock(),
        'reset_button': MagicMock(),
        'logger': MagicMock(),
        'output_log': MagicMock(),
        'ui': MagicMock()
    } 
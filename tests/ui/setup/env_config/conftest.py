"""
File: tests/ui/setup/env_config/conftest.py
Deskripsi: Fixtures for env_config module testing
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

@pytest.fixture
def mock_ui_components():
    """Create a mock UI components dictionary."""
    return {
        'status_bar': MagicMock(),
        'setup_summary': MagicMock(),
        'log_output': MagicMock(),
        '_status_checker': MagicMock(),
        '_logger_bridge': MagicMock()
    }

@pytest.fixture
def env_config_initializer():
    """Create an instance of EnvConfigInitializer for testing."""
    from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
    return EnvConfigInitializer()

@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        'env_name': 'test_env',
        'python_version': '3.10',
        'packages': ['numpy', 'pandas']
    }

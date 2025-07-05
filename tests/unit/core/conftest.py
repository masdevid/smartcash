"""
Shared test fixtures for core module tests.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

# Import the actual modules we need first
import smartcash.ui.core

# Import the mock errors module
import smartcash.tests.unit.core.mock_core_errors

# Import the actual SimpleConfigManager and create a mock instance
from smartcash.common.config.manager import SimpleConfigManager as ConfigManager

# Create a mock config manager for testing
@pytest.fixture
def mock_config_manager():
    """Create a mock config manager for testing."""
    mock = MagicMock(spec=ConfigManager)
    mock.get.return_value = {}
    mock.set.return_value = True
    mock.delete.return_value = True
    return mock

# Setup for core test environment
@pytest.fixture(autouse=True)
def setup_core_test_environment():
    """Set up the test environment for core module tests."""
    # Ensure smartcash.ui.core is in sys.modules
    if 'smartcash.ui.core' not in sys.modules:
        sys.modules['smartcash.ui.core'] = smartcash.ui.core
    
    # Make sure the module is importable
    import smartcash.ui.core.shared.shared_config_manager
    
    yield

# Minimal patching - only patch what's necessary
@pytest.fixture
def mock_config_manager_env(monkeypatch, mock_config_manager):
    """Set up the mock config manager environment."""
    # Patch the get_config_manager function to return our mock
    monkeypatch.setattr('smartcash.common.config.manager.get_config_manager', 
                       lambda: mock_config_manager)
    return mock_config_manager

# Clean up any patches after tests
@pytest.fixture(autouse=True)
def cleanup_after_tests():
    """Clean up after tests."""
    yield
    # Clean up any patches or mocks here if needed

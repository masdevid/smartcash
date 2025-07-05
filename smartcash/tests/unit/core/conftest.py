"""
Test configuration for core module tests.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to Python path if not already there
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our mock config first to patch the imports
from smartcash.tests.unit.core.mock_shared_config import mock_config_manager

@pytest.fixture(autouse=True)
def mock_shared_config():
    """Automatically mock the shared config manager for all tests."""
    with patch('smartcash.ui.core.shared.shared_config_manager.ConfigManager', 
               return_value=mock_config_manager):
        yield mock_config_manager

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch('smartcash.ui.logger') as mock_logger:
        yield mock_logger

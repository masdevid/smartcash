"""
Shared test fixtures for Backbone UI tests.
"""
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_components():
    """Create mock UI components."""
    return {
        'logger_bridge': MagicMock(),
        'model_form': MagicMock(),
        'config_summary': MagicMock()
    }

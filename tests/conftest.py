"""
File: tests/conftest.py
Deskripsi: Konfigurasi dan fixture untuk testing
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create a mock for the dependency services
class MockDependencyServices:
    """Mock for dependency services to avoid actual imports during testing."""
    def __init__(self):
        self.get_dependency_service = MagicMock()
        self.get_dependency_service.return_value = MagicMock()

# Create a mock for the dependency initializer
class MockDependencyInitializer:
    """Mock for dependency initializer to avoid actual imports during testing."""
    def __init__(self):
        pass

@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup mocks for tests."""
    # Create a mock for the dependency module
    mock_dependency = MagicMock()
    mock_dependency.services = MockDependencyServices()
    mock_dependency.dependency_initializer = MockDependencyInitializer()
    
    # Apply the mocks
    with patch.dict('sys.modules', {
        'smartcash.ui.setup.dependency': mock_dependency,
        'smartcash.ui.setup.dependency.services': mock_dependency.services,
        'smartcash.ui.setup.dependency.dependency_initializer': mock_dependency.dependency_initializer,
    }):
        yield

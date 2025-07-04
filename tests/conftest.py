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

# Mock class for dependency initializer
class MockDependencyInitializer:
    """Mock for dependency initializer to avoid actual imports during testing."""
    def __init__(self):
        pass

    def initialize(self):
        return {
            'success': True,
            'ui_components': {'mock_ui': 'component'},
            'config': {'mock_config': True},
            'module_handler': MagicMock(),
            'operation_handlers': {'mock_op': 'handler'}
        }

# Mock class for dependency services
class MockDependencyServices:
    """Mock for dependency services."""
    def __init__(self):
        pass

@pytest.fixture(scope="module")
def optional_dependency_mock():
    """Optional mock for dependency module, only applied when explicitly requested."""
    mock_dependency = MagicMock()
    mock_dependency.services = MockDependencyServices()
    mock_dependency.dependency_initializer = MockDependencyInitializer()
    
    with patch.dict('sys.modules', {
        'smartcash.ui.setup.dependency': mock_dependency,
        'smartcash.ui.setup.dependency.services': mock_dependency.services,
        'smartcash.ui.setup.dependency.dependency_initializer': mock_dependency.dependency_initializer,
    }):
        yield mock_dependency

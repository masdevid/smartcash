"""
Test configuration and setup for the test suite.

This module sets up the test environment, including Python path configuration
and module mocking to ensure tests can run in isolation.
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress specific warnings that are not relevant to tests
warnings.filterwarnings(
    "ignore",
    message="numpy.ufunc size changed",
    category=RuntimeWarning
)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the smartcash package structure
import types
import unittest.mock as mock

# Create mock modules
mock_modules = {
    'smartcash': types.ModuleType('smartcash'),
    'smartcash.ui': types.ModuleType('smartcash.ui'),
    'smartcash.ui.components': types.ModuleType('smartcash.ui.components'),
    'smartcash.ui.core': types.ModuleType('smartcash.ui.core'),
    'smartcash.ui.core.shared': types.ModuleType('smartcash.ui.core.shared'),
    'smartcash.ui.core.handlers': types.ModuleType('smartcash.ui.core.handlers'),
}

# Add mock modules to sys.modules
for name, module in mock_modules.items():
    sys.modules[name] = module

# Now import the mock error handler
from tests.mocks.ui.core.shared.error_handler import ErrorHandler, default_error_handler

# Add the mock error handler to the mock modules
sys.modules['smartcash.ui.core.shared.error_handler'] = types.ModuleType('smartcash.ui.core.shared.error_handler')
sys.modules['smartcash.ui.core.shared.error_handler'].ErrorHandler = ErrorHandler
sys.modules['smartcash.ui.core.shared.error_handler'].default_error_handler = default_error_handler

# Create a mock for ipywidgets
sys.modules['ipywidgets'] = mock.MagicMock()
sys.modules['ipywidgets.widgets'] = mock.MagicMock()

print("✅ Test configuration loaded successfully")

# Import commonly used test utilities for convenience
from unittest.mock import MagicMock, patch, PropertyMock, mock_open  # noqa: E402

# Common test fixtures and utilities can be defined here
__all__ = ['MagicMock', 'patch', 'PropertyMock', 'mock_open']

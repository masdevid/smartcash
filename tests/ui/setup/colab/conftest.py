"""
File: tests/ui/setup/colab/conftest.py
Deskripsi: Konfigurasi test untuk modul colab
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).parents[5]  # Go up 5 levels to reach the project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mock the UI logger to prevent import errors
class MockUILogger:
    """Mock UILogger class for testing."""
    def debug(self, *args, **kwargs):
        pass
    
    def info(self, *args, **kwargs):
        pass
    
    def warning(self, *args, **kwargs):
        pass
    
    def error(self, *args, **kwargs):
        pass
    
    def critical(self, *args, **kwargs):
        pass
    
    def success(self, *args, **kwargs):
        pass

@pytest.fixture(autouse=True)
def mock_ui_dependencies():
    """Mock UI dependencies for testing."""
    # Mock the get_logger function
    with patch('smartcash.ui.utils.ui_logger.get_logger', return_value=MockUILogger()):
        # Mock the UILogger class
        with patch('smartcash.ui.utils.ui_logger.UILogger', return_value=MockUILogger()):
            # Mock the module-level logger
            with patch('smartcash.ui.utils.ui_logger.logging'):
                # Mock the display function from IPython
                with patch('IPython.display.display'):
                    # Mock the HTML class from IPython
                    with patch('IPython.display.HTML'):
                        yield

"""Minimal test file to diagnose pytest issues."""
import os
import sys
import pytest

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

def test_import():
    """Test if we can import the module directly."""
    try:
        from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
        assert True, "Successfully imported FolderHandler"
    except ImportError as e:
        assert False, f"Failed to import FolderHandler: {e}"

def test_import_via_package():
    """Test if we can import via the package."""
    try:
        import smartcash.ui.setup.env_config.handlers.folder_handler
        assert True, "Successfully imported folder_handler module"
    except ImportError as e:
        assert False, f"Failed to import folder_handler module: {e}"

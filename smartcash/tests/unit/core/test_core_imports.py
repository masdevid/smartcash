"""
Test script to verify core imports work correctly.
"""
import pytest
import sys
import importlib
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Core modules to test
CORE_MODULES = [
    "smartcash.ui.core.errors",
    "smartcash.ui.core.handlers.base_handler",
    "smartcash.ui.core.initializers.base_initializer"
]

@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_imports(module_name):
    """Test that core modules can be imported."""
    try:
        module = importlib.import_module(module_name)
        assert module is not None
        print(f"✅ Successfully imported {module_name}")
        print(f"Location: {module.__file__}")
    except (ImportError, AttributeError) as e:
        pytest.fail(f"Failed to import {module_name}: {e}")

# Test specific classes and functions
CORE_IMPORTS = [
    ("smartcash.ui.core.errors", "SmartCashUIError"),
    ("smartcash.ui.core.handlers.base_handler", "BaseHandler"),
    ("smartcash.ui.core.initializers.base_initializer", "BaseInitializer")
]

@pytest.mark.parametrize("module_name,attr_name", CORE_IMPORTS)
def test_core_attributes(module_name, attr_name):
    """Test that core classes and functions can be imported."""
    try:
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        assert attr is not None
        print(f"✅ Successfully imported {module_name}.{attr_name}")
        print(f"Type: {type(attr).__name__}")
    except (ImportError, AttributeError) as e:
        pytest.fail(f"Failed to import {module_name}.{attr_name}: {e}")

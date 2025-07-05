"""
Test core module imports with proper mock configuration.
"""
import sys
import importlib
import pytest
from pathlib import Path

# Core modules to test
CORE_MODULES = [
    "smartcash.ui.core.errors",
    "smartcash.ui.core.handlers.base_handler",
    "smartcash.ui.core.initializers.base_initializer"]

# Test specific classes and functions
CORE_IMPORTS = [
    ("smartcash.ui.core.errors", "SmartCashUIError"),
    ("smartcash.ui.core.handlers.base_handler", "BaseHandler"),
    ("smartcash.ui.core.initializers.base_initializer", "BaseInitializer")]

# Apply our mock configuration first
import smartcash.tests.unit.core.mock_shared_config  # noqa: F401

@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_imports(module_name):
    """Test that core modules can be imported."""
    module = importlib.import_module(module_name)
    assert module is not None
    print(f"✅ Successfully imported {module_name}")
    print(f"Location: {module.__file__}")

@pytest.mark.parametrize("module_name,attr_name", CORE_IMPORTS)
def test_core_attributes(module_name, attr_name):
    """Test that core classes and functions can be imported."""
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    assert attr is not None
    print(f"✅ Successfully imported {module_name}.{attr_name}")
    print(f"Type: {type(attr).__name__}")

def test_imports():
    """Legacy test function for direct execution."""
    for module_name in CORE_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
        print(f"✅ Successfully imported {module_name}")
        print(f"Location: {module.__file__}")
    
    for module_name, attr_name in CORE_IMPORTS:
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        assert attr is not None
        print(f"✅ Successfully imported {module_name}.{attr_name}")
        print(f"Type: {type(attr).__name__}")

if __name__ == "__main__":
    # Add project root to Python path if not already there
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Run the tests
    test_imports()

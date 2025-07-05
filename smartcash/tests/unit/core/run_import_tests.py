"""
Standalone script to test core imports without pytest.
"""
import sys
import importlib
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Apply our mocks first - order is important
print("🔧 Setting up mocks...")
import smartcash.tests.unit.core.mock_core_errors  # noqa: F401 - Must be first
import smartcash.tests.unit.core.mock_shared_config  # noqa: F401

# Core modules to test
CORE_MODULES = [
    "smartcash.ui.core.errors",
    "smartcash.ui.core.handlers.base_handler",
    "smartcash.ui.core.initializers.base_initializer"
]

# Test specific classes and functions
CORE_IMPORTS = [
    ("smartcash.ui.core.errors", "SmartCashUIError"),
    ("smartcash.ui.core.handlers.base_handler", "BaseHandler"),
    ("smartcash.ui.core.initializers.base_initializer", "BaseInitializer")
]

def test_module_import(module_name):
    """Test importing a single module."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        print(f"   Location: {module.__file__}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def test_attribute_import(module_name, attr_name):
    """Test importing a specific attribute from a module."""
    try:
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        print(f"✅ Successfully imported {module_name}.{attr_name}")
        print(f"   Type: {type(attr).__name__}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}.{attr_name}: {e}")
        return False

def main():
    """Run all import tests."""
    print("\n🔍 Testing core module imports...")
    module_results = [test_module_import(module) for module in CORE_MODULES]
    
    print("\n🔍 Testing core class/function imports...")
    attr_results = [test_attribute_import(m, a) for m, a in CORE_IMPORTS]
    
    # Print summary
    total_tests = len(module_results) + len(attr_results)
    passed_tests = sum(module_results) + sum(attr_results)
    
    print("\n📊 Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    
    if passed_tests < total_tests:
        print("\n❌ Some tests failed. See above for details.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()

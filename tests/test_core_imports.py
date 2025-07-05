"""
Test script to verify core imports work correctly.
"""
import sys
import importlib

def test_import(module_name, attr=None):
    """Test importing a module and optionally an attribute."""
    print(f"\n{'='*80}")
    print(f"Testing import: {module_name}" + (f".{attr}" if attr else ""))
    
    try:
        module = importlib.import_module(module_name)
        if attr:
            obj = getattr(module, attr)
            print(f"✅ Successfully imported {module_name}.{attr}")
            print(f"Type: {type(obj).__name__}")
        else:
            print(f"✅ Successfully imported {module_name}")
            print(f"Location: {module.__file__}")
        return True
    except (ImportError, AttributeError) as e:
        print(f"❌ Failed to import {module_name}" + (f".{attr}" if attr else ""))
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Add project root to Python path
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test imports
    test_import("smartcash.ui.core.errors")
    test_import("smartcash.ui.core.errors.error_component")
    test_import("smartcash.ui.core.handlers.base_handler")
    test_import("smartcash.ui.core.initializers.base_initializer")

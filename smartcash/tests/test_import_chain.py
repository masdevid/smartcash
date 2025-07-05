"""
Test script to diagnose import issues in the smartcash package.
"""
import sys
import importlib

def test_import(module_name):
    """Test importing a module and print the result."""
    print(f"\n{'='*80}")
    print(f"Attempting to import: {module_name}")
    print(f"Current sys.path: {sys.path}")
    
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        print(f"Module location: {module.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing {module_name}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Add project root to Python path
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test imports in order of dependency
    modules_to_test = [
        'smartcash',
        'smartcash.ui',
        'smartcash.ui.core',
        'smartcash.ui.core.errors',
        'smartcash.ui.core.errors.error_component',
    ]
    
    all_successful = True
    for module in modules_to_test:
        if not test_import(module):
            all_successful = False
            break
    
    if all_successful:
        print("\n🎉 All imports successful!")
    else:
        print("\n❌ Some imports failed. Check the output above for details.")

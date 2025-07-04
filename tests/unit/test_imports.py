"""
Test script to verify package imports and structure.
"""
import sys
import os
import importlib.util

def test_imports():
    """Test importing the main package and its submodules."""
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    print(f"Python path: {sys.path}")
    
    # Test importing the main package
    try:
        import smartcash
        print("✅ Successfully imported smartcash package")
        if hasattr(smartcash, '__file__'):
            print(f"smartcash.__file__: {smartcash.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import smartcash: {e}")
        return False
    
    # Test importing the UI package
    try:
        from smartcash import ui
        print("✅ Successfully imported smartcash.ui")
        if hasattr(ui, '__file__'):
            print(f"smartcash.ui.__file__: {ui.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import smartcash.ui: {e}")
        return False
    
    # Test importing the setup package
    try:
        from smartcash.ui import setup
        print("✅ Successfully imported smartcash.ui.setup")
        if hasattr(setup, '__file__'):
            print(f"smartcash.ui.setup.__file__: {setup.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import smartcash.ui.setup: {e}")
        return False
    
    # Test importing the colab module
    try:
        from smartcash.ui.setup import colab
        print("✅ Successfully imported smartcash.ui.setup.colab")
        if hasattr(colab, '__file__'):
            print(f"smartcash.ui.setup.colab.__file__: {colab.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import smartcash.ui.setup.colab: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Testing package imports ===")
    success = test_imports()
    print("=== Test completed ===")
    sys.exit(0 if success else 1)

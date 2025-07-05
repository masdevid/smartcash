"""
Debug script to help identify import issues.

This script attempts to import the main container and its dependencies
with debug output to help identify where the import is failing.
"""
import sys
import os
from importlib import import_module

def debug_import(module_name):
    """Attempt to import a module with debug output."""
    print(f"\nAttempting to import: {module_name}")
    try:
        module = import_module(module_name)
        print(f"✅ Successfully imported: {module_name}")
        return module
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        # Print the full traceback for more details
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"❌ Unexpected error importing {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Try importing the main container
    debug_import("smartcash.ui.components.main_container")

"""Test script to verify test imports and package structure."""
import sys
import os
import inspect
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project root: {project_root}")
sys.path.insert(0, project_root)

# Check important directories and files
print("\nChecking directory structure:")
test_dir = os.path.join(project_root, 'tests')
print(f"Test directory: {test_dir} (exists: {os.path.exists(test_dir)})")
ui_test_dir = os.path.join(test_dir, 'ui')
print(f"UI test directory: {ui_test_dir} (exists: {os.path.exists(ui_test_dir)})")

print("\nChecking for __init__.py files:")
for dirpath, dirnames, filenames in os.walk(test_dir):
    if '__init__.py' in filenames:
        print(f"  Found: {os.path.relpath(os.path.join(dirpath, '__init__.py'), project_root)}")

print("\nTrying to import FolderHandler:")
try:
    from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
    print("  Successfully imported FolderHandler!")
    print(f"  Module: {FolderHandler.__module__}")
    print(f"  File: {inspect.getfile(FolderHandler)}")
except ImportError as e:
    print(f"  Error importing FolderHandler: {e}")

print("\nTrying to import test module directly:")
try:
    import importlib.util
    test_file = os.path.join(test_dir, 'ui', 'setup', 'env_config', 'handlers', 'test_folder_handler.py')
    print(f"  Test file: {test_file} (exists: {os.path.exists(test_file)})")
    
    if os.path.exists(test_file):
        spec = importlib.util.spec_from_file_location("test_folder_handler", test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        print("  Successfully loaded test module!")
        
        if hasattr(test_module, 'TestFolderHandler'):
            print("  Found TestFolderHandler class!")
        else:
            print("  Warning: TestFolderHandler class not found in module")
    else:
        print("  Test file not found!")
except Exception as e:
    print(f"  Error loading test module: {e}")

print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

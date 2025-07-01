"""Test script to verify imports."""
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
print(f"Project root: {project_root}")

# Try to import the module
try:
    from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
    print("Successfully imported FolderHandler!")
    print(f"FolderHandler module: {FolderHandler.__module__}")
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPython path:")
    for p in sys.path:
        print(f"  - {p}")
    
    # Check if the file exists
    target_path = os.path.join(project_root, 'smartcash', 'ui', 'setup', 'env_config', 'handlers', 'folder_handler.py')
    print(f"\nChecking if file exists: {target_path}")
    print(f"File exists: {os.path.exists(target_path)}")

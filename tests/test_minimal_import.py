"""
File: tests/test_minimal_import.py
Deskripsi: Minimal test to verify package structure and imports
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Try to import the module directly
    from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
    print("SUCCESS: Successfully imported EnvConfigInitializer")
    print(f"EnvConfigInitializer: {EnvConfigInitializer}")
except ImportError as e:
    print(f"ERROR: Failed to import EnvConfigInitializer: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to find the module manually
    import importlib.util
    
    module_path = os.path.join(project_root, 'smartcash', 'ui', 'setup', 'env_config', 'env_config_initializer.py')
    print(f"Looking for module at: {module_path}")
    print(f"File exists: {os.path.exists(module_path)}")

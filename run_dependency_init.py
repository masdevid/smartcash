"""
File: run_dependency_init.py
Deskripsi: Script untuk menjalankan inisialisasi dependency UI.
"""

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer, initialize_dependency_ui

if __name__ == "__main__":
    print("Attempting to initialize dependency UI directly...")
    try:
        # First try the standard way
        result = initialize_dependency_ui()
        print("Hasil inisialisasi dependency UI (via function):")
        # Since it returns a widget, we can't directly inspect it like a dict
        if result is not None:
            print("UI Widget returned successfully")
        else:
            print("Failed to get UI Widget")
        
        # If you need to inspect the initializer result
        initializer = DependencyInitializer()
        direct_result = initializer.initialize()
        print("\nHasil inisialisasi dependency UI (direct inspection):")
        print(f"Success: {direct_result.get('success', False)}")
        if not direct_result.get('success', False) and 'error' in direct_result:
            print(f"Error: {direct_result['error']}")
        print(f"UI Components: {list(direct_result.get('ui_components', {}).keys()) if direct_result.get('ui_components') else 'None'}")
        print(f"Module Handler: {direct_result.get('module_handler') is not None}")
        print(f"Config Available: {'config' in direct_result}")
        print(f"Operation Handlers: {list(direct_result.get('operation_handlers', {}).keys()) if direct_result.get('operation_handlers') else 'None'}")
    except Exception as e:
        import traceback
        print(f"Unexpected error during initialization: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")

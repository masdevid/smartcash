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
        print(f"Success: {result['success']}")
        if not result['success']:
            for key in result:
                if 'error' in key.lower():
                    print(f"Error ({key}): {result[key]}")
            if 'error' not in result:
                print("Error: No specific error key found in result")
                print(f"Full result dictionary: {result}")
        print(f"UI Components: {list(result['ui_components'].keys()) if result['ui_components'] else 'None'}")
        print(f"Module Handler: {result['module_handler'] is not None}")
        print(f"Config Available: {'config' in result}")
        print(f"Operation Handlers: {list(result['operation_handlers'].keys()) if result['operation_handlers'] else 'None'}")
        
        # If it fails, try direct initialization for more control
        if not result['success']:
            print("\nAttempting direct initialization of DependencyInitializer...")
            initializer = DependencyInitializer()
            direct_result = initializer.initialize()
            print("Hasil inisialisasi dependency UI (direct):")
            print(f"Success: {direct_result['success']}")
            if not direct_result['success'] and 'error' in direct_result:
                print(f"Error: {direct_result['error']}")
            print(f"UI Components: {list(direct_result['ui_components'].keys()) if direct_result['ui_components'] else 'None'}")
            print(f"Module Handler: {direct_result['module_handler'] is not None}")
            print(f"Config Available: {'config' in direct_result}")
            print(f"Operation Handlers: {list(direct_result['operation_handlers'].keys()) if direct_result['operation_handlers'] else 'None'}")
    except Exception as e:
        import traceback
        print(f"Unexpected error during initialization: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")

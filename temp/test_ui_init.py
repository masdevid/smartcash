# Test script to validate the initialization of the preprocessing UI.

import sys
import os

# Add project root to the Python path to allow for correct module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"Project root added to path: {project_root}")

try:
    # Import the initializer function from the correct module
    from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import initialize_preprocessing_ui
    print("Successfully imported initialize_preprocessing_ui.")
    
    # Call the function to set up the UI module
    module = initialize_preprocessing_ui(display=False) # display=False to prevent widget rendering in test
    print("Successfully initialized preprocessing UI module.")
    
    # A simple check to ensure the main container widget exists
    main_container = module.get_component('main_container')
    if main_container:
        print("Validation successful: 'main_container' widget found.")
    else:
        print("Validation failed: 'main_container' not found in UI components.")

except ImportError as e:
    print(f"Import Error: {e}. Please check module paths and dependencies.")
except Exception as e:
    print(f"An unexpected error occurred during UI initialization: {e}")

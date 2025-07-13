"""
Test script to debug UI display issues with backbone UI.
"""
from IPython.display import display, clear_output
import ipywidgets as widgets

# Import the backbone UI module
from smartcash.ui.model.backbone.backbone_uimodule import initialize_backbone_ui

def test_ui_display():
    """Test function to display the backbone UI."""
    # Clear any existing output
    clear_output(wait=True)
    
    # Initialize the UI with display=False to get the components
    result = initialize_backbone_ui(display=False)
    
    if result and 'ui_components' in result:
        ui_components = result['ui_components']
        
        # Try different ways to display the UI
        if 'main_container' in ui_components:
            main_ui = ui_components['main_container']
            print("\n=== Attempting to display main_container ===")
            
            # Method 1: Direct display
            print("\nMethod 1: Direct display")
            display(main_ui)
            
            # Method 2: Show method if available
            if hasattr(main_ui, 'show'):
                print("\nMethod 2: Using show() method")
                display(main_ui.show())
            
            # Method 3: Access container attribute
            if hasattr(main_ui, 'container'):
                print("\nMethod 3: Accessing .container")
                display(main_ui.container)
            
            # Method 4: Try to get the underlying widget
            if hasattr(main_ui, '_ipython_display_'):
                print("\nMethod 4: Using _ipython_display_")
                display(main_ui)
            
            # Check for 'ui' reference
            if 'ui' in ui_components and ui_components['ui'] is not main_ui:
                print("\nMethod 5: Using 'ui' reference from components")
                display(ui_components['ui'])
    
    return result

# Run the test
if __name__ == "__main__":
    test_ui_display()

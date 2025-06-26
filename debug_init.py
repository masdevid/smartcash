#!/usr/bin/env python3
"""Debug script for ConfigCellInitializer initialization."""

import sys
import traceback
import ipywidgets as widgets
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer

def main():
    print("=== Starting debug script ===")
    
    try:
        # Create a mock parent component with VBox widgets
        print("\n1. Creating mock parent component...")
        parent_component = type('MockParent', (), {})()
        parent_component.container = widgets.VBox()
        parent_component.content_area = widgets.VBox()
        
        print("2. Creating ConfigCellInitializer instance...")
        initializer = ConfigCellInitializer(
            module_name='test_module',
            config_filename='test_config.yaml',
            parent_component=parent_component
        )
        
        print("\n3. Before initialize() call:")
        print(f"  - initializer type: {type(initializer).__name__}")
        print(f"  - parent_component.container type: {type(parent_component.container).__name__}")
        print(f"  - parent_component.content_area type: {type(parent_component.content_area).__name__}")
        
        # Call initialize
        print("\n4. Calling initialize()...")
        result = initializer.initialize()
        
        print("\n5. After initialize() call:")
        print(f"  - result type: {type(result).__name__}")
        print(f"  - result is None: {result is None}")
        
        # Check if it's a widget
        is_widget = hasattr(result, '_repr_mimebundle_') or hasattr(result, '_ipython_display_')
        print(f"\n6. Widget check:")
        print(f"  - Is result a widget? {is_widget}")
        
        if not is_widget and result is not None:
            print("\n7. Available attributes:")
            print(f"  - {dir(result)[:20]}...")
        
        print("\n=== Debug script completed successfully ===")
        
    except Exception as e:
        print("\n!!! ERROR !!!")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

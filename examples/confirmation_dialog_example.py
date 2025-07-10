"""
Example demonstrating the usage of ConfirmationDialog and SimpleDialog.

This example shows how to use both the legacy ConfirmationDialog and the new
SimpleDialog to create and display various types of dialogs.
"""

import ipywidgets as widgets
from IPython.display import display

# Import the dialog components
from smartcash.ui.components.dialog.confirmation_dialog import (
    ConfirmationDialog,
    create_confirmation_area,
    show_confirmation_dialog,
    show_info_dialog
)
from smartcash.ui.components.dialog.simple_dialog import (
    SimpleDialog,
    create_simple_dialog,
    show_confirmation_dialog as simple_show_confirmation_dialog,
    show_info_dialog as simple_show_info_dialog,
    show_success_dialog,
    show_warning_dialog,
    show_error_dialog
)

def create_demo_ui():
    """Create a demo UI showing different ways to use dialogs."""
    # Create output widget for displaying messages
    output = widgets.Output()
    
    # Create a container for the UI
    ui_components = {}
    
    # --- Legacy ConfirmationDialog ---
    legacy_header = widgets.HTML(
        value="<h3>Legacy ConfirmationDialog</h3>"
             "<p>These examples use the legacy ConfirmationDialog API.</p>"
    )
    
    # Button to show legacy confirmation dialog
    legacy_confirm_btn = widgets.Button(
        description="Show Legacy Confirmation",
        button_style="primary"
    )
    
    def on_legacy_confirm_clicked(b):
        clear_output()
        with output:
            print("Showing legacy confirmation dialog...")
        
        def on_confirm():
            with output:
                print("✅ User confirmed the operation!")
        
        def on_cancel():
            with output:
                print("❌ User cancelled the operation")
        
        # Using the legacy show_confirmation_dialog function
        show_confirmation_dialog(
            ui_components,
            title="Legacy Confirmation",
            message="This is a legacy confirmation dialog.\nDo you want to proceed?",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Yes, Proceed",
            cancel_text="No, Cancel",
            danger_mode=True
        )
    
    legacy_confirm_btn.on_click(on_legacy_confirm_clicked)
    
    # --- SimpleDialog ---
    simple_header = widgets.HTML(
        value="<h3 style='margin-top: 20px;'>New SimpleDialog</h3>"
             "<p>These examples use the new SimpleDialog API.</p>"
    )
    
    # Create a SimpleDialog instance
    simple_dialog = create_simple_dialog("example_dialog")
    
    # Button to show simple confirmation dialog
    simple_confirm_btn = widgets.Button(
        description="Show Simple Confirmation",
        button_style="primary"
    )
    
    def on_simple_confirm_clicked(b):
        clear_output()
        with output:
            print("Showing simple confirmation dialog...")
        
        def on_confirm():
            with output:
                print("✅ User confirmed the operation!")
        
        def on_cancel():
            with output:
                print("❌ User cancelled the operation")
        
        # Using the simple_show_confirmation_dialog function
        simple_show_confirmation_dialog(
            dialog=simple_dialog,
            title="Simple Confirmation",
            message="This is a simple confirmation dialog.\nDo you want to proceed?",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Yes, Proceed",
            cancel_text="No, Cancel",
            danger_mode=False
        )
    
    simple_confirm_btn.on_click(on_simple_confirm_clicked)
    
    # Button to show success dialog
    success_btn = widgets.Button(
        description="Show Success Dialog",
        button_style="success"
    )
    
    def on_success_clicked(b):
        clear_output()
        with output:
            print("Showing success dialog...")
        
        show_success_dialog(
            dialog=simple_dialog,
            title="Operation Successful",
            message="Your operation was completed successfully!",
            on_ok=lambda: output.append_stdout("✅ Success dialog closed\n")
        )
    
    success_btn.on_click(on_success_clicked)
    
    # Button to show error dialog
    error_btn = widgets.Button(
        description="Show Error Dialog",
        button_style="danger"
    )
    
    def on_error_clicked(b):
        clear_output()
        with output:
            print("Showing error dialog...")
        
        show_error_dialog(
            dialog=simple_dialog,
            title="Error Occurred",
            message="An unexpected error occurred while processing your request.",
            on_ok=lambda: output.append_stdout("❌ Error dialog closed\n")
        )
    
    error_btn.on_click(on_error_clicked)
    
    # Assemble the UI
    legacy_section = widgets.VBox([
        legacy_header,
        legacy_confirm_btn,
        widgets.HTML("<div style='margin: 10px 0; border-top: 1px solid #eee;'></div>")
    ])
    
    simple_section = widgets.VBox([
        simple_header,
        simple_confirm_btn,
        success_btn,
        error_btn
    ])
    
    # Create the main UI
    ui = widgets.VBox([
        widgets.HTML("<h2>Dialog Component Demo</h2>"),
        legacy_section,
        simple_section,
        widgets.HTML("<h4>Output:</h4>"),
        output
    ])
    
    # Initialize the legacy confirmation area
    create_confirmation_area(ui_components)
    
    return ui


if __name__ == "__main__":
    from IPython.display import clear_output
    
    print("🎭 Dialog Component Examples")
    print("=" * 50)
    print("This example demonstrates the usage of both legacy ConfirmationDialog")
    print("and the new SimpleDialog components.")
    print("\nTo run this example in a notebook:")
    print("1. Create a new notebook")
    print("2. Copy the code from this file")
    print("3. Run it in a notebook cell")
    print("\nOr run it as a script to see this help message.")

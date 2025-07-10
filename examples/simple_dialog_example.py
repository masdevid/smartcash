"""
Example usage of the SimpleDialog component.

This example demonstrates how to use the simplified dialog component
with basic hide/show functionality.
"""

import ipywidgets as widgets
from IPython.display import display

# Import the new simple dialog components
from smartcash.ui.components.dialog import (
    create_simple_dialog,
    simple_show_confirmation_dialog,
    simple_show_info_dialog,
    show_success_dialog,
    show_warning_dialog,
    show_error_dialog
)


def create_dialog_demo():
    """Create a demonstration of the simple dialog component."""
    
    # Create a dialog instance
    dialog = create_simple_dialog("demo_dialog")
    
    # Create output widget to show messages
    output = widgets.Output()
    
    # Create buttons to trigger different dialog types
    confirm_button = widgets.Button(
        description="Show Confirmation",
        button_style="primary",
        layout=widgets.Layout(margin="5px")
    )
    
    info_button = widgets.Button(
        description="Show Info",
        button_style="info",
        layout=widgets.Layout(margin="5px")
    )
    
    success_button = widgets.Button(
        description="Show Success",
        button_style="success",
        layout=widgets.Layout(margin="5px")
    )
    
    warning_button = widgets.Button(
        description="Show Warning",
        button_style="warning",
        layout=widgets.Layout(margin="5px")
    )
    
    error_button = widgets.Button(
        description="Show Error",
        button_style="danger",
        layout=widgets.Layout(margin="5px")
    )
    
    danger_button = widgets.Button(
        description="Show Danger Confirmation",
        button_style="danger",
        layout=widgets.Layout(margin="5px")
    )
    
    clear_button = widgets.Button(
        description="Clear Dialog",
        button_style="",
        layout=widgets.Layout(margin="5px")
    )
    
    # Define callback functions
    def on_confirm_clicked(b):
        def on_confirm():
            with output:
                print("✅ User confirmed the operation!")
        
        def on_cancel():
            with output:
                print("🚫 User cancelled the operation")
        
        simple_show_confirmation_dialog(
            dialog=dialog,
            title="Confirm Operation",
            message="Are you sure you want to proceed with this operation?",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Yes, Proceed",
            cancel_text="Cancel"
        )
    
    def on_info_clicked(b):
        def on_ok():
            with output:
                print("ℹ️ Info dialog closed")
        
        simple_show_info_dialog(
            dialog=dialog,
            title="Information",
            message="This is an informational message to provide context about the current operation.",
            on_ok=on_ok,
            ok_text="Got it",
            info_type="info"
        )
    
    def on_success_clicked(b):
        def on_ok():
            with output:
                print("✅ Success dialog closed")
        
        show_success_dialog(
            dialog=dialog,
            title="Operation Successful",
            message="The operation has been completed successfully!",
            on_ok=on_ok,
            ok_text="Great!"
        )
    
    def on_warning_clicked(b):
        def on_ok():
            with output:
                print("⚠️ Warning dialog closed")
        
        show_warning_dialog(
            dialog=dialog,
            title="Warning",
            message="Please be careful with the next operation. It may have side effects.",
            on_ok=on_ok,
            ok_text="Understood"
        )
    
    def on_error_clicked(b):
        def on_ok():
            with output:
                print("❌ Error dialog closed")
        
        show_error_dialog(
            dialog=dialog,
            title="Error Occurred",
            message="An unexpected error occurred during the operation. Please try again.",
            on_ok=on_ok,
            ok_text="OK"
        )
    
    def on_danger_clicked(b):
        def on_confirm():
            with output:
                print("🔥 Dangerous operation confirmed!")
        
        def on_cancel():
            with output:
                print("🛡️ Dangerous operation cancelled")
        
        simple_show_confirmation_dialog(
            dialog=dialog,
            title="⚠️ Dangerous Operation",
            message="This operation will permanently delete all data. This action cannot be undone!",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Yes, Delete All",
            cancel_text="Cancel",
            danger_mode=True
        )
    
    def on_clear_clicked(b):
        dialog.clear()
        with output:
            print("🧹 Dialog cleared")
    
    # Bind event handlers
    confirm_button.on_click(on_confirm_clicked)
    info_button.on_click(on_info_clicked)
    success_button.on_click(on_success_clicked)
    warning_button.on_click(on_warning_clicked)
    error_button.on_click(on_error_clicked)
    danger_button.on_click(on_danger_clicked)
    clear_button.on_click(on_clear_clicked)
    
    # Create the UI layout
    ui = widgets.VBox([
        widgets.HTML("<h2>🎭 Simple Dialog Demo</h2>"),
        widgets.HTML("<p>Click the buttons below to see different dialog types in action:</p>"),
        
        widgets.HBox([
            confirm_button,
            info_button,
            success_button
        ]),
        
        widgets.HBox([
            warning_button,
            error_button,
            danger_button
        ]),
        
        widgets.HBox([
            clear_button
        ]),
        
        widgets.HTML("<h3>📋 Dialog Area</h3>"),
        dialog.container,  # This is where dialogs will appear
        
        widgets.HTML("<h3>📝 Output Log</h3>"),
        output
    ])
    
    return ui


def create_practical_example():
    """Create a practical example showing dialog usage in a data processing context."""
    
    # Create dialog
    dialog = create_simple_dialog("data_processor_dialog")
    
    # Create output widget
    output = widgets.Output()
    
    # Create UI components
    file_selector = widgets.Text(
        placeholder="Enter file path...",
        description="File Path:",
        style={'description_width': '100px'},
        layout=widgets.Layout(width='300px')
    )
    
    process_button = widgets.Button(
        description="Process File",
        button_style="primary",
        layout=widgets.Layout(margin="5px")
    )
    
    delete_button = widgets.Button(
        description="Delete File",
        button_style="danger",
        layout=widgets.Layout(margin="5px")
    )
    
    # Simulate file processing
    def process_file():
        file_path = file_selector.value
        if not file_path:
            show_error_dialog(
                dialog=dialog,
                title="❌ No File Selected",
                message="Please enter a file path before processing.",
                on_ok=lambda: None
            )
            return
        
        # Simulate processing
        import random
        success = random.choice([True, False])
        
        if success:
            show_success_dialog(
                dialog=dialog,
                title="✅ Processing Complete",
                message=f"File '{file_path}' has been processed successfully!",
                on_ok=lambda: output.append_stdout(f"✅ Processed: {file_path}\n")
            )
        else:
            show_error_dialog(
                dialog=dialog,
                title="❌ Processing Failed",
                message=f"Failed to process file '{file_path}'. Please check the file and try again.",
                on_ok=lambda: output.append_stdout(f"❌ Failed: {file_path}\n")
            )
    
    def delete_file():
        file_path = file_selector.value
        if not file_path:
            show_warning_dialog(
                dialog=dialog,
                title="⚠️ No File Selected",
                message="Please enter a file path before deleting.",
                on_ok=lambda: None
            )
            return
        
        def confirm_delete():
            # Simulate deletion
            show_success_dialog(
                dialog=dialog,
                title="🗑️ File Deleted",
                message=f"File '{file_path}' has been deleted successfully.",
                on_ok=lambda: output.append_stdout(f"🗑️ Deleted: {file_path}\n")
            )
            file_selector.value = ""  # Clear the input
        
        def cancel_delete():
            output.append_stdout(f"🚫 Delete cancelled: {file_path}\n")
        
        simple_show_confirmation_dialog(
            dialog=dialog,
            title="🗑️ Confirm Deletion",
            message=f"Are you sure you want to delete '{file_path}'? This action cannot be undone.",
            on_confirm=confirm_delete,
            on_cancel=cancel_delete,
            confirm_text="Yes, Delete",
            cancel_text="Cancel",
            danger_mode=True
        )
    
    # Bind events
    process_button.on_click(lambda b: process_file())
    delete_button.on_click(lambda b: delete_file())
    
    # Create UI layout
    ui = widgets.VBox([
        widgets.HTML("<h2>📁 File Processor Example</h2>"),
        widgets.HTML("<p>This example shows how to use dialogs in a practical file processing application:</p>"),
        
        widgets.HBox([
            file_selector,
            process_button,
            delete_button
        ]),
        
        widgets.HTML("<h3>📋 Dialog Area</h3>"),
        dialog.container,
        
        widgets.HTML("<h3>📝 Processing Log</h3>"),
        output
    ])
    
    return ui


if __name__ == "__main__":
    print("🎭 Simple Dialog Component Examples")
    print("=" * 50)
    
    print("\n1. Basic Dialog Demo:")
    demo_ui = create_dialog_demo()
    display(demo_ui)
    
    print("\n2. Practical File Processing Example:")
    practical_ui = create_practical_example()
    display(practical_ui)
    
    print("\n✨ Examples created successfully!")
    print("Click the buttons above to interact with the dialogs.")
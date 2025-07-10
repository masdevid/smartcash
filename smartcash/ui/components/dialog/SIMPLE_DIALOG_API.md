# 📋 SimpleDialog API Documentation

## 🎯 Overview

The `SimpleDialog` component provides a clean, simplified interface for displaying confirmation and info dialogs without complex animations, JavaScript, or CSS. It uses basic ipywidgets functionality with auto-expanding/hiding confirmation areas.

## 🚀 Key Features

- **Simple Hide/Show**: Basic visibility control without animations
- **Auto-expanding**: Confirmation area automatically expands when shown, hides when not needed
- **Multiple Dialog Types**: Confirmation, info, success, warning, and error dialogs
- **Clean API**: Easy-to-use factory functions
- **Error Handling**: Built-in error handling for callbacks
- **Non-breaking**: Maintains backward compatibility with legacy components

## 📋 Basic Usage

### Creating a Dialog

```python
from smartcash.ui.components.dialog import create_simple_dialog

# Create a dialog instance
dialog = create_simple_dialog("my_dialog")

# Display the dialog container in your UI
display(dialog.container)
```

### Confirmation Dialog

```python
from smartcash.ui.components.dialog import show_confirmation_dialog

def on_confirm():
    print("✅ User confirmed!")
    # Execute your operation here

def on_cancel():
    print("🚫 User cancelled")

show_confirmation_dialog(
    dialog=dialog,
    title="Delete Files",
    message="Are you sure you want to delete all files?",
    on_confirm=on_confirm,
    on_cancel=on_cancel,
    confirm_text="Yes, Delete",
    cancel_text="Cancel",
    danger_mode=True
)
```

### Info Dialogs

```python
from smartcash.ui.components.dialog import (
    show_info_dialog,
    show_success_dialog,
    show_warning_dialog,
    show_error_dialog
)

def on_ok():
    print("ℹ️ Dialog closed")

# Basic info dialog
show_info_dialog(
    dialog=dialog,
    title="Information",
    message="Operation completed successfully!",
    on_ok=on_ok,
    ok_text="Got it",
    info_type="info"
)

# Convenience functions for specific types
show_success_dialog(dialog, "Success", "Files processed successfully!")
show_warning_dialog(dialog, "Warning", "Please check your settings")
show_error_dialog(dialog, "Error", "Something went wrong")
```

## 📋 API Reference

### `SimpleDialog` Class

#### Constructor
```python
SimpleDialog(component_name: str = "simple_dialog", **kwargs)
```

#### Methods

##### `show_confirmation(title, message, on_confirm=None, on_cancel=None, confirm_text="Confirm", cancel_text="Cancel", danger_mode=False)`
Show a confirmation dialog with confirm and cancel buttons.

**Parameters:**
- `title: str` - Dialog title
- `message: str` - Dialog message
- `on_confirm: Callable` - Callback for confirm button
- `on_cancel: Callable` - Callback for cancel button
- `confirm_text: str` - Text for confirm button (default: "Confirm")
- `cancel_text: str` - Text for cancel button (default: "Cancel")
- `danger_mode: bool` - Use danger styling (default: False)

##### `show_info(title, message, on_ok=None, ok_text="OK", info_type="info")`
Show an info dialog with a single OK button.

**Parameters:**
- `title: str` - Dialog title
- `message: str` - Dialog message
- `on_ok: Callable` - Callback for OK button
- `ok_text: str` - Text for OK button (default: "OK")
- `info_type: str` - Type of info dialog: "info", "success", "warning", "error"

##### `hide()`
Hide the dialog and clean up state.

##### `clear()`
Clear the dialog content and hide it.

##### `is_visible() -> bool`
Check if the dialog is currently visible.

### Factory Functions

#### `create_simple_dialog(component_name: str = "dialog") -> SimpleDialog`
Create and initialize a simple dialog instance.

#### `show_confirmation_dialog(dialog, title, message, on_confirm=None, on_cancel=None, confirm_text="Confirm", cancel_text="Cancel", danger_mode=False)`
Show a confirmation dialog using a SimpleDialog instance.

#### `show_info_dialog(dialog, title, message, on_ok=None, ok_text="OK", info_type="info")`
Show an info dialog using a SimpleDialog instance.

#### `show_success_dialog(dialog, title, message, on_ok=None, ok_text="OK")`
Show a success dialog (green styling).

#### `show_warning_dialog(dialog, title, message, on_ok=None, ok_text="OK")`
Show a warning dialog (yellow styling).

#### `show_error_dialog(dialog, title, message, on_ok=None, ok_text="OK")`
Show an error dialog (red styling).

## 🎨 Styling

The SimpleDialog uses basic ipywidgets styling with color-coded borders:

- **Normal**: Blue border (`#007bff`)
- **Danger**: Red border (`#dc3545`)
- **Success**: Green border (`#28a745`)
- **Warning**: Yellow border (`#ffc107`)
- **Error**: Red border (`#dc3545`)

## 🔧 Integration Examples

### With Operation Handlers

```python
from smartcash.ui.components.dialog import create_simple_dialog, show_confirmation_dialog

class DataProcessor:
    def __init__(self):
        self.dialog = create_simple_dialog("processor_dialog")
        # Add dialog.container to your UI
        
    def delete_files(self):
        show_confirmation_dialog(
            dialog=self.dialog,
            title="🗑️ Delete Files",
            message="This will permanently delete all processed files. Continue?",
            on_confirm=self._execute_delete,
            on_cancel=self._cancel_delete,
            confirm_text="Yes, Delete",
            cancel_text="Cancel",
            danger_mode=True
        )
    
    def _execute_delete(self):
        try:
            # Perform deletion
            self._show_success("Files deleted successfully!")
        except Exception as e:
            self._show_error(f"Failed to delete files: {str(e)}")
    
    def _cancel_delete(self):
        print("🚫 Delete operation cancelled")
    
    def _show_success(self, message):
        show_success_dialog(
            self.dialog,
            "✅ Success",
            message,
            on_ok=lambda: print("Success acknowledged")
        )
    
    def _show_error(self, message):
        show_error_dialog(
            self.dialog,
            "❌ Error",
            message,
            on_ok=lambda: print("Error acknowledged")
        )
```

### With UI Components

```python
from smartcash.ui.components.dialog import create_simple_dialog, show_info_dialog
import ipywidgets as widgets

def create_ui():
    # Create dialog
    dialog = create_simple_dialog("ui_dialog")
    
    # Create button to trigger dialog
    button = widgets.Button(description="Show Info")
    
    def on_button_click(b):
        show_info_dialog(
            dialog=dialog,
            title="Button Clicked",
            message="You clicked the button!",
            on_ok=lambda: print("Info dialog closed"),
            info_type="info"
        )
    
    button.on_click(on_button_click)
    
    # Create UI layout
    ui = widgets.VBox([
        widgets.HTML("<h3>Simple Dialog Example</h3>"),
        button,
        dialog.container  # Add dialog container to UI
    ])
    
    return ui

# Usage
ui = create_ui()
display(ui)
```

## 🔄 Migration from Legacy Dialog

### Before (Legacy)
```python
from smartcash.ui.components.dialog import show_confirmation_dialog

# Legacy usage with ui_components dict
ui_components = {}
show_confirmation_dialog(
    ui_components=ui_components,
    title="Confirm",
    message="Are you sure?",
    on_confirm=confirm_callback
)
```

### After (Simple Dialog)
```python
from smartcash.ui.components.dialog import create_simple_dialog, simple_show_confirmation_dialog

# New usage with dialog instance
dialog = create_simple_dialog("my_dialog")
simple_show_confirmation_dialog(
    dialog=dialog,
    title="Confirm",
    message="Are you sure?",
    on_confirm=confirm_callback
)
```

## 🛠️ Best Practices

### 1. Dialog Instance Management
```python
# ✅ Good - Create dialog once, reuse multiple times
dialog = create_simple_dialog("app_dialog")

def show_delete_confirmation():
    show_confirmation_dialog(dialog, "Delete", "Are you sure?", ...)

def show_success_message():
    show_success_dialog(dialog, "Success", "Operation completed!")
```

### 2. Error Handling
```python
# ✅ Good - Handle callback errors gracefully
def safe_operation():
    try:
        # Perform operation
        show_success_dialog(dialog, "Success", "Operation completed!")
    except Exception as e:
        show_error_dialog(dialog, "Error", f"Operation failed: {str(e)}")
```

### 3. Clear State
```python
# ✅ Good - Clear dialog when switching contexts
def switch_mode():
    dialog.clear()  # Clear any existing dialog
    # Continue with new mode
```

### 4. UI Layout
```python
# ✅ Good - Always include dialog container in UI
def create_ui():
    dialog = create_simple_dialog("ui_dialog")
    
    ui = widgets.VBox([
        # Your UI components
        widgets.HTML("<h3>Main Content</h3>"),
        
        # Dialog container (initially hidden)
        dialog.container
    ])
    
    return ui, dialog
```

## 🔍 Troubleshooting

### Dialog Not Showing
- Ensure `dialog.container` is added to your UI layout
- Check that the dialog is properly initialized
- Verify no exceptions in callback functions

### Styling Issues
- Use `info_type` parameter for different dialog types
- Set `danger_mode=True` for destructive actions
- Content HTML supports basic styling

### State Management
- Use `dialog.clear()` to reset dialog state
- Check `dialog.is_visible()` to verify current state
- Callbacks automatically hide the dialog after execution

## 📦 Export Summary

```python
from smartcash.ui.components.dialog import (
    # Main class
    SimpleDialog,
    
    # Factory function
    create_simple_dialog,
    
    # Dialog functions
    simple_show_confirmation_dialog,  # Avoid conflict with legacy
    simple_show_info_dialog,          # Avoid conflict with legacy
    show_success_dialog,
    show_warning_dialog,
    show_error_dialog
)
```
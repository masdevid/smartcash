# Dialog Components

Dialog components provide modal interfaces for user interactions such as confirmations, forms, and detailed information display.

## Components

### ConfirmationDialog (`confirmation_dialog.py`)
A modal dialog that requires user confirmation before proceeding with an action.

**Props:**
- `title` (str): Dialog title
- `message` (str): Main message to display
- `on_confirm` (callable): Function to call when confirmed
- `on_cancel` (callable, optional): Function to call when cancelled
- `confirm_text` (str, optional): Text for confirm button (default: "Confirm")
- `cancel_text` (str, optional): Text for cancel button (default: "Cancel")
- `is_open` (bool): Controls dialog visibility
- `on_open_change` (callable): Callback when dialog visibility changes

**Example:**
```python
from smartcash.ui.components.dialog.confirmation_dialog import ConfirmationDialog

def handle_confirm():
    print("Action confirmed!")

def handle_cancel():
    print("Action cancelled!")

ConfirmationDialog(
    title="Delete Item",
    message="Are you sure you want to delete this item?",
    on_confirm=handle_confirm,
    on_cancel=handle_cancel,
    confirm_text="Delete",
    cancel_text="Keep"
)
```

## Best Practices

- Use for critical actions that require user confirmation
- Keep dialog content concise and focused
- Make the primary action clear and distinct
- Support keyboard navigation (Escape to cancel, Enter to confirm)
- Ensure proper focus management when dialog opens/closes

## Accessibility

- Implements ARIA modal dialog pattern
- Manages focus trapping
- Provides keyboard navigation support
- Includes proper ARIA labels and roles
- Ensures sufficient color contrast

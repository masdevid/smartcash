# Alert Components

Alert components are used to display important messages to users, such as success messages, warnings, errors, and informational notices.

## Components

### Alert (`alert.py`)
Base alert component that can be extended for different alert types.

**Props:**
- `message` (str): The message to display
- `type` (str): One of 'success', 'error', 'warning', 'info'
- `dismissible` (bool): Whether the alert can be dismissed
- `on_dismiss` (callable): Callback when alert is dismissed

**Example:**
```python
from smartcash.ui.components.alerts.alert import Alert

Alert(
    message="Operation completed successfully!",
    type="success",
    dismissible=True
)
```

### InfoBox (`info_box.py`)
Styled container for displaying informational content.

**Props:**
- `title` (str): Title of the info box
- `content` (str): Main content
- `icon` (str, optional): Icon to display

### StatusIndicator (`status_indicator.py`)
Visual indicator showing the status of a process or system.

**Props:**
- `status` (str): Current status ('idle', 'loading', 'success', 'error')
- `message` (str): Status message to display
- `show_spinner` (bool): Whether to show loading spinner

## Best Practices

- Use appropriate alert types for different message severities
- Keep alert messages clear and concise
- Use dismissible alerts for non-critical messages
- Include relevant actions when appropriate (e.g., "Retry" for errors)
- Ensure sufficient color contrast for accessibility

## Accessibility

- All alerts include appropriate ARIA roles and attributes
- Color is not the only indicator of status
- Focus management is handled automatically for dismissible alerts

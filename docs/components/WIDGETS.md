# Form Widgets

Interactive form elements for user input and data collection.

## Components

### TextInput (`widgets/text_input.py`)
Basic text input field.

**Props:**
- `value` (str): Current value
- `placeholder` (str, optional): Placeholder text
- `disabled` (bool): Whether the input is disabled
- `type` (str): Input type ('text', 'password', 'email', etc.)
- `on_change` (callable): Callback when value changes
- `error` (str, optional): Error message to display

**Example:**
```python
from smartcash.ui.components.widgets.text_input import TextInput

def handle_change(value):
    print(f"Input changed: {value}")

text_input = TextInput(
    placeholder="Enter your name",
    on_change=handle_change
)
```

### Checkbox (`widgets/checkbox.py`)
Checkbox input for boolean values.

**Props:**
- `checked` (bool): Whether checked
- `label` (str): Label text
- `disabled` (bool): Whether disabled
- `on_change` (callable): Callback when checked state changes

### Dropdown (`widgets/dropdown.py`)
Select dropdown with options.

**Props:**
- `options` (List[Dict]): Available options
- `value`: Currently selected value
- `label` (str, optional): Field label
- `on_change` (callable): Callback when selection changes
- `multiple` (bool): Allow multiple selection

**Example:**
```python
from smartcash.ui.components.widgets.dropdown import Dropdown

options = [
    {"value": "option1", "label": "Option 1"},
    {"value": "option2", "label": "Option 2"}
]

dropdown = Dropdown(
    options=options,
    value="option1",
    label="Select an option"
)
```

### Slider (`widgets/slider.py`)
Range slider for numeric input.

**Props:**
- `min` (float): Minimum value
- `max` (float): Maximum value
- `value` (float): Current value
- `step` (float): Increment step
- `on_change` (callable): Callback when value changes
- `marks` (Dict, optional): Markers on the slider

### LogSlider (`widgets/log_slider.py`)
Logarithmic slider for wide value ranges.

**Props:**
- `min` (float): Minimum value (log scale)
- `max` (float): Maximum value (log scale)
- `value` (float): Current value
- `base` (float): Logarithm base (default: 10)
- `on_change` (callable): Callback when value changes

## Action Buttons

### ActionButtons (`action_buttons.py`)
Container for action buttons.

**Props:**
- `actions` (List[Dict]): List of button definitions
- `align` (str): Alignment ('left', 'center', 'right')
- `variant` (str): Button style variant

### SaveResetButtons (`save_reset_buttons.py`)
Standard save and reset button pair.

**Props:**
- `on_save` (callable): Save button callback
- `on_reset` (callable): Reset button callback
- `save_disabled` (bool): Whether save is disabled
- `reset_disabled` (bool): Whether reset is disabled
- `save_text` (str): Save button text
- `reset_text` (str): Reset button text

## Best Practices

- Always provide clear labels
- Use appropriate input types
- Implement proper validation
- Provide helpful error messages
- Support keyboard navigation
- Ensure touch targets are large enough
- Group related form elements

## Form Handling

### Controlled Components
All form widgets are controlled components, meaning they don't maintain their own internal state. The parent component is responsible for managing the state.

**Example:**
```python
class MyForm(Component):
    def __init__(self):
        self.state = {"name": "", "email": ""}
    
    def handle_change(self, field, value):
        self.state[field] = value
        self.update()
    
    def render(self):
        return f"""
            <div class="form">
                {TextInput(
                    value=self.state["name"],
                    on_change=lambda v: self.handle_change("name", v),
                    placeholder="Name"
                )}
                {TextInput(
                    value=self.state["email"],
                    on_change=lambda v: self.handle_change("email", v),
                    placeholder="Email",
                    type="email"
                )}
            </div>
        """
```

## Validation

### Basic Validation
```python
def validate_email(email):
    if "@" not in email:
        return "Please enter a valid email address"
    return None

email_input = TextInput(
    value="",
    on_change=handle_email_change,
    error=validate_email(email)
)
```

### Form-Level Validation
```python
def validate_form(data):
    errors = {}
    if not data.get("name"):
        errors["name"] = "Name is required"
    if not data.get("email"):
        errors["email"] = "Email is required"
    elif "@" not in data["email"]:
        errors["email"] = "Invalid email format"
    return errors
```

## Accessibility

- All form controls have proper labels
- Error messages are associated with their inputs
- Required fields are clearly marked
- Keyboard navigation is fully supported
- ARIA attributes are used appropriately
- Color contrast meets accessibility standards

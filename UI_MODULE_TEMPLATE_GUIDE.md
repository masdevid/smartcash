# SmartCash UI Module Template Guide

## Overview

This guide provides a comprehensive template for creating standardized UI modules in the SmartCash application. The template ensures consistency across all modules while allowing for customization based on specific requirements.

## Template Structure

The template follows the exact container order specified in the UI module structure documentation:

1. **Header Container** (Header + Status Panel)
2. **Form Container** (Custom to each module)
3. **Action Container** (Save/Reset | Primary | Action Buttons)
4. **Summary Container** (Custom, Nice to have)
5. **Operation Container** (Progress + Dialog + Log)
6. **Footer Container** (Info Accordion + Tips)

## Quick Start

### 1. Copy and Rename Template

```bash
# Copy the template to your module directory
cp ui_module_template.py smartcash/ui/[parent_module]/[module_name]/components/[module_name]_ui.py

# Example:
cp ui_module_template.py smartcash/ui/dataset/preprocess/components/preprocess_ui.py
```

### 2. Update Module Information

Replace the following placeholders in the copied file:

```python
# Replace [module] with your actual module name
def create_[module]_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

# Example:
def create_preprocess_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
```

### 3. Configure Module Constants

Update the `UI_CONFIG` dictionary:

```python
UI_CONFIG = {
    'title': "Data Preprocessing",  # Module display title
    'subtitle': "Clean and prepare dataset for training",  # Module description
    'icon': "🧹",  # Module icon
    'module_name': "preprocess",  # Module name
    'parent_module': "dataset",  # Parent module name
    'version': "1.0.0"  # Module version
}
```

### 4. Configure Action Buttons

Update the `BUTTON_CONFIG` dictionary:

```python
BUTTON_CONFIG = {
    'process': {
        'text': '🧹 Process Data',
        'style': 'primary',
        'tooltip': 'Start data preprocessing',
        'order': 1
    },
    'validate': {
        'text': '✅ Validate',
        'style': 'info',
        'tooltip': 'Validate preprocessed data',
        'order': 2
    },
    'export': {
        'text': '📤 Export',
        'style': 'success',
        'tooltip': 'Export processed data',
        'order': 3
    }
}
```

### 5. Implement Form Widgets

Customize the `_create_module_form_widgets()` function:

```python
def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create preprocessing-specific form widgets."""
    
    # Image resize settings
    resize_enabled = widgets.Checkbox(
        value=config.get('resize_enabled', True),
        description='Enable Image Resizing',
        style={'description_width': 'initial'}
    )
    
    target_size = widgets.IntText(
        value=config.get('target_size', 640),
        description='Target Size:',
        style={'description_width': '120px'}
    )
    
    # Normalization settings
    normalize_enabled = widgets.Checkbox(
        value=config.get('normalize_enabled', True),
        description='Enable Normalization',
        style={'description_width': 'initial'}
    )
    
    # Create form layout
    form_ui = widgets.VBox([
        widgets.HTML("<h4>🧹 Preprocessing Configuration</h4>"),
        resize_enabled,
        target_size,
        normalize_enabled
    ])
    
    return {
        'ui': form_ui,
        'resize_enabled': resize_enabled,
        'target_size': target_size,
        'normalize_enabled': normalize_enabled
    }
```

## Detailed Customization Guide

### Container Configuration

#### Header Container
- Always required
- Contains module title, subtitle, and status
- Status updates automatically based on operations

#### Form Container
- Fully customizable per module
- Use `_create_module_form_widgets()` to define form elements
- Support for validation rules

#### Action Container
- **Save/Reset Buttons**: Include when module needs persistence config
- **Primary Button**: Use for single main operation only
- **Action Buttons**: Use for multiple operations
- **Rule**: Never use Primary Button with Action Buttons simultaneously

#### Summary Container
- Optional but recommended
- Display module status, statistics, or preview
- Customize `_create_module_summary_content()`

#### Operation Container
- Always required
- Contains progress tracker, dialog area, and log accordion
- Provides functions for handlers: `log_message`, `update_progress`, `show_dialog`

#### Footer Container
- Optional but recommended
- Contains info accordion and tips
- Customize `_create_module_info_box()` and `_create_module_tips_box()`

### Button Configuration Guidelines

#### Single Operation Module
```python
action_container = create_action_container(
    buttons=[],  # No action buttons
    show_save_reset=True,  # If persistence needed
    primary_button_config={
        'text': '🚀 Start Processing',
        'style': 'primary',
        'tooltip': 'Start main operation'
    }
)
```

#### Multiple Operations Module
```python
action_container = create_action_container(
    buttons=[
        {'id': 'validate', 'text': '✅ Validate', 'style': 'info', 'order': 1},
        {'id': 'process', 'text': '🧹 Process', 'style': 'primary', 'order': 2},
        {'id': 'export', 'text': '📤 Export', 'style': 'success', 'order': 3}
    ],
    show_save_reset=True,  # If persistence needed
    primary_button_config=None  # No primary button
)
```

### Validation Rules

Define validation rules for form fields:

```python
VALIDATION_RULES = {
    'target_size': {
        'min': 224,
        'max': 1024,
        'required': True
    },
    'batch_size': {
        'min': 1,
        'max': 64,
        'required': True
    }
}
```

### Error Handling

The template includes error handling through decorators:

```python
@handle_ui_errors(error_component_title="[Module] UI Creation Error")
def create_[module]_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    # UI creation logic
```

## Return Value Structure

The template returns a standardized dictionary with the following keys:

### Required Keys
- `'ui'`: Main UI widget (VBox containing all containers)
- `'header_container'`: Header container widget
- `'form_container'`: Form container widget
- `'action_container'`: Action container widget
- `'operation_container'`: Operation container widget
- `'footer_container'`: Footer container widget

### Optional Keys
- `'summary_container'`: Summary container widget (if enabled)
- `'primary_button'`: Primary action button
- `'save_button'`: Save button (if save/reset enabled)
- `'reset_button'`: Reset button (if save/reset enabled)
- `'[action]_button'`: Individual action buttons

### Widget References
- `'form_widgets'`: Dictionary containing form widgets
- `'log_message'`: Function to log messages
- `'update_progress'`: Function to update progress
- `'show_dialog'`: Function to show dialogs
- `'progress_tracker'`: Progress tracker widget
- `'log_output'`: Log output widget
- `'log_accordion'`: Log accordion widget

### Metadata
- `'module_name'`: Module name
- `'parent_module'`: Parent module name
- `'logger_namespace'`: Logger namespace
- `'ui_initialized'`: Initialization flag
- `'config'`: Module configuration
- `'version'`: Module version

## Integration with Handlers

The template is designed to work seamlessly with module handlers:

```python
# In your module's handler
class PreprocessHandler:
    def __init__(self):
        self.ui_components = create_preprocess_ui()
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        # Access buttons through standardized keys
        if self.ui_components.get('process_button'):
            self.ui_components['process_button'].on_click(self._on_process_click)
        
        if self.ui_components.get('save_button'):
            self.ui_components['save_button'].on_click(self._on_save_click)
    
    def _on_process_click(self, button):
        # Use standardized functions
        self.ui_components['log_message']("Starting preprocessing...", "info")
        self.ui_components['update_progress'](0, "Initializing...")
        
        # Your processing logic here
        
        self.ui_components['update_progress'](100, "Complete!")
```

## Best Practices

### 1. Module Structure
```
[module_name]/
├── components/
│   ├── __init__.py
│   ├── [module_name]_ui.py      # Main UI (use template)
│   ├── form_widgets.py          # Custom form widgets
│   └── summary_widgets.py       # Custom summary widgets
├── constants.py                 # Module constants
├── handlers/
│   └── [module_name]_handler.py # Module handler
└── __init__.py
```

### 2. Imports Organization
```python
# Standard library imports
import ipywidgets as widgets
from typing import Dict, Any, Optional

# Core UI imports (standardized)
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
# ... other container imports

# Module-specific imports
from .form_widgets import create_custom_form
from ..constants import UI_CONFIG, BUTTON_CONFIG
```

### 3. Configuration Management
```python
# Always provide defaults
config = config or get_module_default_config()

# Validate configuration
config = validate_module_config(config)

# Use configuration throughout
widget_value = config.get('field_name', default_value)
```

### 4. Error Handling
```python
# Use error handling decorators
@handle_ui_errors(error_component_title="Module UI Creation Error")
def create_module_ui(config):
    # UI creation logic
    pass

# Validate inputs
def validate_module_config(config):
    # Validation logic
    if not config.get('required_field'):
        raise ValueError("Required field is missing")
    return config
```

## Common Patterns

### Pattern 1: Single Operation Module
- Use primary button for main action
- Include save/reset if persistence needed
- Simple form with basic options

### Pattern 2: Multiple Operations Module
- Use action buttons for different operations
- Include save/reset if persistence needed
- More complex form with multiple sections

### Pattern 3: Analysis Module
- Include summary container for results
- Use charts or visualizations
- Multiple action buttons for different analyses

### Pattern 4: Configuration Module
- Focus on form container
- Include validation and preview
- Save/reset buttons for persistence

## Testing

Test your UI module with:

```python
# Basic functionality test
def test_module_ui_creation():
    ui_components = create_module_ui()
    assert 'ui' in ui_components
    assert 'header_container' in ui_components
    assert ui_components['ui_initialized'] is True

# Configuration test
def test_module_ui_with_config():
    config = {'field_name': 'test_value'}
    ui_components = create_module_ui(config)
    assert ui_components['config'] == config

# Button presence test
def test_module_buttons():
    ui_components = create_module_ui()
    assert 'primary_button' in ui_components
    assert ui_components['primary_button'] is not None
```

## Migration from Existing UI

To migrate existing UI modules to use this template:

1. **Backup existing module**
2. **Analyze current UI structure**
3. **Map existing widgets to template containers**
4. **Update configuration and constants**
5. **Test thoroughly**
6. **Update handlers if needed**

This template provides a solid foundation for creating consistent, maintainable UI modules across the SmartCash application while allowing for the flexibility needed for module-specific requirements.
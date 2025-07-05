# SmartCash Container System

This document provides an overview of the container components used in the SmartCash UI system. The container system provides a flexible way to organize and manage the layout of UI components.

## Table of Contents
- [Overview](#overview)
- [Container Types](#container-types)
- [Main Container](#main-container)
  - [Creating a Main Container](#creating-a-main-container)
  - [Container Configuration](#container-configuration)
- [Container Components](#container-components)
  - [Header Container](#header-container)
  - [Form Container](#form-container)
  - [Action Container](#action-container)
  - [Operation Container](#operation-container)
  - [Footer Container](#footer-container)
  - [Summary Container](#summary-container)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Overview

The SmartCash container system provides a structured way to organize UI components. The system is built around the `MainContainer` component, which serves as the root container for all UI elements. The `MainContainer` can contain multiple child containers, each serving a specific purpose.

## Container Types

The following container types are available:

| Type | Description |
|------|-------------|
| `header` | Container for header content with navigation |
| `form` | Container for form elements with consistent styling |
| `action` | Container for action buttons and controls |
| `operation` | Container for operation-related components (progress, dialogs, logs) |
| `footer` | Container for footer content |
| `custom` | Custom container type for specialized use cases |

## Main Container

The `MainContainer` is the root container that holds all other containers. It provides flexible ordering and visibility control for child containers.

### Creating a Main Container

You can create a main container in two ways:

1. **Legacy Way**: Pass individual container components

```python
from smartcash.ui.components import create_main_container

container = create_main_container(
    header_container=header,
    form_container=form,
    action_container=actions,
    operation_container=operations,
    footer_container=footer
)
```

2. **New Flexible Way**: Pass a list of component configurations

```python
from smartcash.ui.components import create_main_container

components = [
    {'type': 'header', 'component': header_widget, 'order': 0},
    {'type': 'form', 'component': form_widget, 'order': 1},
    {'type': 'action', 'component': actions_widget, 'order': 2},
    {'type': 'operation', 'component': operations_widget, 'order': 3},
    {'type': 'footer', 'component': footer_widget, 'order': 4}
]

container = create_main_container(components=components)
```

### Container Configuration

Each container configuration can have the following properties:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | `ContainerType` | Yes | The type of container (header, form, action, etc.) |
| `component` | `widgets.Widget` | Yes | The widget to display in this container |
| `order` | `int` | No | The display order (lower numbers appear first) |
| `name` | `str` | No | An optional name for the container |
| `visible` | `bool` | No | Whether the container is visible (default: True) |

## Container Components

### Header Container

The `HeaderContainer` is used to display header content such as titles, subtitles, and status messages.

```python
from smartcash.ui.components import create_header_container

header = create_header_container(
    title="Application Title",
    subtitle="Manage your application settings",
    status_message="Ready",
    status_type="info"
)
```

### Form Container

The `FormContainer` provides a consistent way to organize form elements with proper spacing and styling.

```python
from smartcash.ui.components import create_form_container

form = create_form_container(
    layout_type="column",  # or "row", "grid"
    container_margin="0",
    container_padding="10px",
    gap="10px"
)

# Add form elements
form['add_item'](input_widget, height="auto")
```

### Action Container

The `ActionContainer` is designed to hold action buttons and controls.

```python
from smartcash.ui.components import create_action_container

actions = create_action_container(
    buttons=[
        {
            'button_id': 'submit',
            'text': 'Submit',
            'style': 'primary',
            'order': 1
        },
        {
            'button_id': 'cancel',
            'text': 'Cancel',
            'style': 'secondary',
            'order': 2
        }
    ],
    title="Actions",
    alignment="center"
)
```

### Operation Container

The `OperationContainer` is used to display operation-related components such as progress indicators, dialogs, and logs.

```python
from smartcash.ui.components import create_operation_container

operations = create_operation_container(
    show_progress=True,
    show_dialog=True,
    show_logs=True
)
```

### Footer Container

The `FooterContainer` is used to display footer content such as version information and links.

```python
from smartcash.ui.components import create_footer_container

footer = create_footer_container(
    panels=[
        {
            'panel_type': 'info_accordion',
            'title': 'ℹ️ Information',
            'content': '<p>Additional information goes here</p>',
            'style': 'info',
            'flex': '1',
            'min_width': '300px',
            'open_by_default': True
        }
    ],
    style={"border_top": "1px solid #e0e0e0", "padding": "10px 0"},
    flex_flow="row wrap",
    justify_content="space-between"
)
```

### Summary Container

The `SummaryContainer` is used to display summary information or dashboard widgets.

```python
from smartcash.ui.components import create_summary_container

summary = create_summary_container(
    title="Summary",
    items=[
        {'label': 'Total Items', 'value': '42', 'icon': '📊'},
        {'label': 'Completed', 'value': '25', 'icon': '✅'},
        {'label': 'In Progress', 'value': '12', 'icon': '⏳'},
        {'label': 'Failed', 'value': '5', 'icon': '❌'}
    ]
)
```

## Best Practices

1. **Use the new flexible container system** with explicit ordering for new code.
2. **Maintain backward compatibility** when updating existing code.
3. **Keep container types consistent** with their intended purposes.
4. **Use meaningful names** for custom containers.
5. **Set appropriate visibility** for containers that should be hidden by default.
6. **Use the factory functions** (`create_*_container`) instead of direct class instantiation.

## Examples

### Creating a Complete UI

```python
from smartcash.ui.components import (
    create_main_container,
    create_header_container,
    create_form_container,
    create_action_container,
    create_operation_container,
    create_footer_container
)

# Create individual containers
header = create_header_container(
    title="SmartCash",
    subtitle="Manage your finances",
    status_message="Ready",
    status_type="info"
)

form = create_form_container(layout_type="column")
form['add_item'](input_widget1, height="auto")
form['add_item'](input_widget2, height="auto")

actions = create_action_container(
    buttons=[
        {'button_id': 'save', 'text': '💾 Save', 'style': 'primary'},
        {'button_id': 'reset', 'text': '🔄 Reset', 'style': 'secondary'}
    ]
)

operations = create_operation_container(
    show_progress=True,
    show_dialog=True,
    show_logs=True
)

footer = create_footer_container(
    panels=[
        {
            'panel_type': 'info_accordion',
            'title': 'ℹ️ About',
            'content': '<p>SmartCash v1.0.0</p>'
        }
    ]
)

# Combine into main container
components = [
    {'type': 'header', 'component': header, 'order': 0},
    {'type': 'form', 'component': form, 'order': 1},
    {'type': 'action', 'component': actions, 'order': 2},
    {'type': 'operation', 'component': operations, 'order': 3},
    {'type': 'footer', 'component': footer, 'order': 4}
]

ui = create_main_container(components=components)
```

### Updating Container Visibility

```python
# Hide a container
ui.hide_container('operation')

# Show a container
ui.show_container('operation')

# Toggle container visibility
ui.toggle_container_visibility('operation')
```

### Getting a Container by Type

```python
# Get a container by type
form_container = ui.get_container('form')

# Update the form container
form_container['add_item'](new_widget, height="auto")
```

This documentation reflects the current implementation as of the latest update. For more details, refer to the source code of each container component.

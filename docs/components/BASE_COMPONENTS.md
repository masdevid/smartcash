# Base Components

Core components that serve as building blocks for the application UI.

## Components

### BaseComponent (`base_component.py`)
Abstract base class for all UI components.

**Methods:**
- `render()`: Abstract method to render the component
- `update(**kwargs)`: Update component properties
- `add_class(className)`: Add a CSS class
- `remove_class(className)`: Remove a CSS class

### Card (`card.py`)
Container component with a card-like appearance.

**Props:**
- `title` (str, optional): Card title
- `subtitle` (str, optional): Card subtitle
- `elevation` (int): Shadow depth (0-24)
- `padding` (str): CSS padding value
- `margin` (str): CSS margin value

**Example:**
```python
from smartcash.ui.components.card import Card

card = Card(
    title="User Profile",
    subtitle="View and edit profile information",
    elevation=2
)
```

### Tabs (`tabs.py`)
Tabbed interface component.

**Props:**
- `tabs` (List[Dict]): List of tab definitions
- `active_tab` (str): Currently active tab ID
- `on_tab_change` (callable): Callback when tab changes

**Example:**
```python
tabs = [
    {"id": "profile", "label": "Profile"},
    {"id": "settings", "label": "Settings"},
    {"id": "billing", "label": "Billing"}
]

def handle_tab_change(tab_id):
    print(f"Switched to tab: {tab_id}")

tabs_component = Tabs(
    tabs=tabs,
    active_tab="profile",
    on_tab_change=handle_tab_change
)
```

## Panel Components

### InfoPanel (`info_panel.py`)
Panel for displaying informational content.

### StatusPanel (`status_panel.py`)
Panel for displaying system/process status.

### SummaryPanel (`summary_panel.py`)
Panel for displaying summary information.

### TipsPanel (`tips_panel.py`)
Panel for displaying helpful tips.

### CloseableTipsPanel (`closeable_tips_panel.py`)
Tips panel that can be closed by the user.

## Best Practices

- Use semantic HTML elements when possible
- Maintain consistent spacing and padding
- Ensure proper keyboard navigation
- Implement proper ARIA attributes
- Follow the application's design system

## Common Patterns

### Creating a New Component
```python
from smartcash.ui.components.base_component import BaseComponent

class MyComponent(BaseComponent):
    def __init__(self, **props):
        super().__init__(**props)
        self.add_class("my-component")
    
    def render(self):
        return f"""
            <div class="{self.classes}">
                <!-- Component content -->
            </div>
        """
```

### Using Containers
```python
from smartcash.ui.components import (
    MainContainer,
    Card,
    Tabs
)

# Create a main container with a card and tabs
container = MainContainer()
card = Card(title="My Application")
tabs = Tabs(tabs=[...])

# Add components to the card
card.add_component(tabs)
container.add_component(card)
```

## Accessibility

- All components include proper ARIA attributes
- Keyboard navigation is fully supported
- Color contrast meets WCAG standards
- Focus management is handled automatically
- Screen reader support is implemented

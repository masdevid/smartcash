# Shared UI Components

This directory contains reusable UI components used throughout the SmartCash application. Each component is designed to be modular, maintainable, and consistent with the application's design system.

## Component Categories

1. [Alerts](#alerts)
2. [Dialog](#dialog)
3. [Error Handling](#error-handling)
4. [Header](#header)
5. [Info Components](#info-components)
6. [Layout](#layout)
7. [Logging](#logging)
8. [Progress Tracking](#progress-tracking)
9. [Widgets](#widgets)

## Alerts

Location: `components/alerts/`

Components for displaying various types of alerts and notifications to users.

- **Alert** (`alert.py`): Base alert component
- **InfoBox** (`info_box.py`): For displaying informational messages
- **StatusIndicator** (`status_indicator.py`): Visual indicator of system/process status
- **StatusUpdater** (`status_updater.py`): Manages and updates status indicators

## Dialog

Location: `components/dialog/`

Reusable dialog components for user interactions.

- **ConfirmationDialog** (`confirmation_dialog.py`): Modal dialog for confirming user actions

## Error Handling

Location: `components/error/`

Components for displaying and managing error states.

- **ErrorComponent** (`error_component.py`): Displays error messages to users

## Header

Location: `components/header/`

Header-related components for the application.

- **Header** (`header.py`): Main application header component

## Info Components

Location: `components/info/`

Components for displaying information to users.

- **InfoComponent** (`info_component.py`): Base component for information display

## Layout

Location: `components/layout/`

Layout components for structuring the application UI.

- **LayoutComponents** (`layout_components.py`): Collection of common layout components

## Logging

Location: `components/log_accordion/`

Components for displaying and managing application logs.

- **LogAccordion** (`log_accordion.py`): Collapsible log viewer
- **LogEntry** (`log_entry.py`): Individual log entry component
- **LogLevel** (`log_level.py`): Log level definitions and utilities

## Progress Tracking

Location: `components/progress_tracker/`

Components for tracking and displaying progress of operations.

- **ProgressTracker** (`progress_tracker.py`): Main progress tracking component
- **CallbackManager** (`callback_manager.py`): Manages progress callbacks
- **TqdmManager** (`tqdm_manager.py`): Integration with tqdm progress bars
- **UIComponents** (`ui_components.py`): UI elements for progress tracking

## Widgets

Location: `components/widgets/`

Various reusable UI widgets.

## Usage Guidelines

1. **Importing Components**:
   ```python
   from smartcash.ui.components.alerts.alert import Alert
   from smartcash.ui.components.dialog.confirmation_dialog import ConfirmationDialog
   ```

2. **Component Props**:
   Each component's props and their types are documented in the component's docstring.

3. **Styling**:
   Components follow the application's design system. Use the provided props for styling rather than custom CSS when possible.

4. **Accessibility**:
   All components are built with accessibility in mind. Ensure to provide appropriate ARIA attributes when necessary.

## Best Practices

- Always check component documentation for required props
- Use the most specific component available for your use case
- Keep component usage consistent throughout the application
- Test components in different states (loading, error, success, etc.)
- Follow the principle of least privilege when passing props

## Contributing

When adding new components:

1. Place them in the appropriate category directory
2. Include comprehensive docstrings
3. Add TypeScript type definitions if applicable
4. Include usage examples in the component's docstring
5. Update this documentation if adding a new category

## Testing

Components should include unit tests in the corresponding `__tests__` directory. Run tests with:

```bash
pytest tests/ui/components/
```

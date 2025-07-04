# UI Shared Components

## Overview

This document outlines the shared UI components used across the SmartCash application. These components are designed to be reusable, consistent, and follow the project's design system.

## Core Component Categories

### 1. Base Components
Fundamental building blocks used throughout the application.
- Reference: [Base Components Documentation](../components/BASE_COMPONENTS.md)

### 2. Widgets
Interactive UI elements for user interaction.
- Reference: [Widgets Documentation](../components/WIDGETS.md)

### 3. Dialogs
Modal dialogs and popups for focused user interactions.
- Reference: [Dialog Documentation](../components/DIALOG.md)

### 4. Alerts & Notifications
User feedback components for status updates and important messages.
- Reference: [Alerts Documentation](../components/ALERTS.md)

### 5. Progress Tracking
Components for displaying operation progress and status.
- Reference: [Progress Tracker Documentation](../components/PROGRESS_TRACKER.md)

### 6. Logging
Components for displaying and managing application logs.
- Reference: [Logging Documentation](../components/LOGGING.md)

## Implementation Guidelines

### Component Naming
- Use PascalCase for component names (e.g., `ProgressBar`)
- Prefix with module name for module-specific components (e.g., `DatasetProgressBar`)

### Props & State
- Document all props with TypeScript interfaces
- Use descriptive prop names that indicate their purpose
- Group related props together in the interface

### Styling
- Use CSS modules for component-specific styles
- Follow the project's design tokens for colors, spacing, and typography
- Ensure responsive behavior for different screen sizes

### Accessibility
- Include proper ARIA attributes
- Ensure keyboard navigation support
- Provide text alternatives for non-text content

## Best Practices

1. **Composition Over Inheritance**
   - Build complex components by composing simpler ones
   - Keep components small and focused on a single responsibility

2. **State Management**
   - Lift state up when multiple components need access to the same data
   - Use context for global state that many components need

3. **Performance**
   - Memoize expensive calculations
   - Implement `shouldComponentUpdate` or use `React.memo` for performance optimization
   - Lazy load components that aren't immediately needed

4. **Testing**
   - Write unit tests for component logic
   - Include accessibility tests
   - Add visual regression tests for critical components

## Versioning & Updates

When updating shared components:
1. Document breaking changes
2. Provide migration guides when necessary
3. Maintain backward compatibility when possible
4. Update the relevant documentation

## Component Library

For a complete reference of all available components and their usage, see the [Components Documentation](../components/README.md).

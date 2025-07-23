# Enhanced Button State Management

## Overview

The button handler mixin has been enhanced to provide comprehensive button state management that accommodates module-specific requirements while maintaining consistency across the entire UI system.

## Key Features

### 1. **Dependency-Based Button States**
Buttons can now have dependency functions that determine when they should be enabled/disabled:

```python
# Set a dependency check function
self.set_button_dependency('validate', self._check_validate_dependency)

def _check_validate_dependency(self) -> bool:
    """Return True if validate button should be enabled"""
    return self.has_required_models()
```

### 2. **Reason Tracking**
When buttons are disabled, the system tracks the reason:

```python
# Disable with reason
self.disable_button('build', reason="Missing data prerequisites")

# Get the reason later
reason = self.get_button_disable_reason('build')
```

### 3. **Conditional Visibility**
Update multiple button states based on conditions:

```python
# Update multiple buttons based on conditions
button_conditions = {
    'validate': has_models,
    'build': has_data,
    'export': has_results
}

button_reasons = {
    'validate': "No models available" if not has_models else None,
    'build': "No data available" if not has_data else None,
    'export': "No results to export" if not has_results else None
}

self.update_button_states_based_on_condition(button_conditions, button_reasons)
```

### 4. **Enhanced State Queries**
Check button states and reasons:

```python
# Check if button is enabled
if self.is_button_enabled('validate'):
    # Button is ready to use
    pass

# Get disable reason
reason = self.get_button_disable_reason('validate')
if reason:
    self.log(f"Validate button disabled: {reason}")
```

## Implementation in Backbone Module

The backbone module demonstrates proper usage of the enhanced button state management:

### Dependency Setup
```python
def _setup_backbone_button_dependencies(self) -> None:
    """Setup button dependencies for backbone-specific requirements."""
    # Validate button depends on having built models
    self.set_button_dependency('validate', self._check_validate_button_dependency)
    
    # Build button depends on data prerequisites  
    self.set_button_dependency('build', self._check_build_button_dependency)

def _check_validate_button_dependency(self) -> bool:
    """Check if validate button should be enabled."""
    if not self._operation_service:
        return False
        
    # Quick check: if we have any built models, enable validate
    result = self._operation_service.rescan_built_models()
    return result.get('success', False) and result.get('total_models', 0) > 0
```

### State Updates
```python
def _update_ui_from_scan_results(self, scan_result: Dict[str, Any]) -> None:
    """Update UI based on scan results."""
    has_models = scan_result.get('total_models', 0) > 0
    
    # Update button states using enhanced mixin functionality
    button_conditions = {
        'validate': has_models,
    }
    
    button_reasons = {
        'validate': "No built models available" if not has_models else None,
    }
    
    self.update_button_states_based_on_condition(button_conditions, button_reasons)
```

## Benefits

### 1. **Consistency**
All modules use the same button state management system, ensuring consistent behavior across the UI.

### 2. **Maintainability**
Button logic is centralized in the mixin, reducing code duplication and making it easier to enhance or debug.

### 3. **Flexibility**
Modules can define custom dependency checks and state update logic while still using the standardized infrastructure.

### 4. **Debugging**
Reason tracking makes it easy to understand why buttons are in certain states, improving user experience and debugging.

### 5. **Extensibility**
New button state features can be added to the mixin and immediately available to all modules.

## Migration Guide

For existing modules, replace custom button state logic:

### Before (Custom Implementation):
```python
def _update_button_visibility(self, button_id: str, visible: bool):
    # Custom logic to find and update button
    action_container = self._ui_components.get('action_container')
    if action_container:
        buttons = action_container.get('buttons', {})
        button = buttons.get(button_id)
        if button:
            button.disabled = not visible
```

### After (Enhanced Mixin):
```python
def _update_button_visibility(self, button_id: str, visible: bool):
    reason = "Prerequisites not met" if not visible else None
    self.set_button_visibility(button_id, visible, reason)
```

## Best Practices

1. **Define Dependencies Early**: Set up button dependencies during module initialization
2. **Use Descriptive Reasons**: Provide clear reasons when disabling buttons
3. **Batch Updates**: Use `update_button_states_based_on_condition()` for multiple buttons
4. **Check Dependencies**: Let the mixin handle dependency checking automatically
5. **Consistent Naming**: Use standard button IDs across modules

This enhanced system provides a robust, consistent, and maintainable approach to managing button states across all UI modules.
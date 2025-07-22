# Error Handling Migration Guide

This document provides guidance for migrating from the old `error_handler.py` to the new modular error handling system in `smartcash.ui.core.errors`.

## Overview

The error handling system has been refactored into a more modular and maintainable structure. The main changes include:

1. **Modular Organization**: Error handling functionality is now split across multiple focused modules.
2. **Enhanced Type Hints**: Better type safety and IDE support.
3. **Improved Documentation**: More comprehensive docstrings and examples.
4. **New Features**: Additional utilities for common error handling patterns.

## Migration Steps

### 1. Update Imports

#### Old Import
```python
from smartcash.ui.core.shared.error_handler import CoreErrorHandler, ErrorLevel
```

#### New Import
```python
from smartcash.ui.core.errors import CoreErrorHandler, ErrorLevel
```

### 2. Error Handling Patterns

#### Basic Error Handling

**Old Way:**
```python
try:
    # Code that might fail
    result = some_operation()
except Exception as e:
    error_handler.handle_exception(e, "Operation failed")
```

**New Way:**
```python
from smartcash.ui.core.errors import with_error_handling

@with_error_handling("Operation failed")
def perform_operation():
    return some_operation()

result = perform_operation()
```

### 3. Error Context

**Old Way:**
```python
try:
    # Code with context
    with some_context() as ctx:
        error_handler.set_context(context_id=ctx.id)
        operation()
finally:
    error_handler.clear_context()
```

**New Way:**
```python
from smartcash.ui.core.errors import ErrorContext

with ErrorContext.context(context_id=ctx.id):
    operation()
```

### 4. Input Validation

**Old Way:**
```python
if not isinstance(value, str):
    error_handler.handle_error("Invalid type", level=ErrorLevel.ERROR)
```

**New Way:**
```python
from smartcash.ui.core.errors.validators import validate_type

validate_type(value, str, "value")
```

### 5. Safe Operations

**Old Way:**
```python
try:
    result = component.some_operation()
except Exception as e:
    error_handler.handle_exception(e, "Component operation failed")
    result = None
```

**New Way:**
```python
from smartcash.ui.core.errors.validators import safe_component_operation

result = safe_component_operation(
    component, 
    "some_operation",
    component_name="MyComponent"
)
```

## New Features

### Retry Mechanism

```python
from smartcash.ui.core.errors.decorators import retry_on_failure

@retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data():
    # Will automatically retry on failure
    return requests.get("https://api.example.com/data").json()
```

### Error Context Manager

```python
from smartcash.ui.core.errors import ErrorContext

with ErrorContext.context(request_id=request_id, user=current_user):
    # All errors in this block will include the context
    process_request()
```

### Detailed Error Information

```python
from smartcash.ui.core.errors.utils import get_exception_info

try:
    risky_operation()
except Exception as e:
    error_info = get_exception_info(e)
    logger.error("Operation failed", extra={"error_info": error_info})
```

## Backward Compatibility

The new module maintains backward compatibility with the old `CoreErrorHandler` API. However, it's recommended to update to the new patterns for better maintainability.

## Testing

After migration, thoroughly test all error handling paths to ensure they work as expected with the new implementation.

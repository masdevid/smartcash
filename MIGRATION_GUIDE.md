# Migration Guide: Moving from `smartcash.ui.handlers` to `smartcash.ui.core`

This guide provides instructions for migrating your code to use the new `smartcash.ui.core` module structure. The old `smartcash.ui.handlers` and `smartcash.ui.initializers` modules are now deprecated and will be removed in a future version.

## Changes Overview

| Old Import | New Import | Notes |
|------------|------------|-------|
| `smartcash.ui.handlers.error_handler` | `smartcash.ui.core.errors.handlers` | Error handling functionality has been moved to the `core.errors` package |
| `smartcash.ui.handlers.base_handler` | `smartcash.ui.core.handlers` | Base handler functionality has been moved to `core.handlers` |
| `smartcash.ui.handlers.config_handlers` | `smartcash.ui.core.config` | Config handler functionality has been moved to `core.config` |
| `smartcash.ui.initializers` | `smartcash.ui.core.initializers` | Initializers have been moved to the core package |

## Detailed Migration Steps

### 1. Update Imports

#### Error Handling

Before:
```python
from smartcash.ui.handlers.error_handler import (
    handle_ui_errors,
    safe_execute,
    create_error_response,
    ErrorContext
)
```

After:
```python
from smartcash.ui.core.errors.handlers import (
    handle_ui_errors,
    safe_execute,
    create_error_response,
    ErrorContext
)
```

#### Base Handler

Before:
```python
from smartcash.ui.handlers.base_handler import BaseHandler
```

After:
```python
from smartcash.ui.core.handlers import BaseHandler
```

#### Config Handlers

Before:
```python
from smartcash.ui.handlers.config_handlers import (
    ConfigHandler,
    create_config_handler,
    get_or_create_handler
)
```

After:
```python
from smartcash.ui.core.config import (
    ConfigHandler,
    create_config_handler,
    get_or_create_handler
)
```

### 2. Update Error Component Imports

If you were using the error component directly:

Before:
```python
from smartcash.ui.components.error.error_component import (
    ErrorComponent,
    create_error_component
)
```

After:
```python
from smartcash.ui.core.errors.error_component import (
    ErrorComponent,
    create_error_component
)
```

### 3. Update Initializer Imports

Before:
```python
from smartcash.ui.initializers import (
    initialize_ui_components,
    setup_event_handlers
)
```

After:
```python
from smartcash.ui.core.initializers import (
    initialize_ui_components,
    setup_event_handlers
)
```

## Deprecation Warnings

If you see deprecation warnings like:
```
DeprecationWarning: The 'smartcash.ui.handlers' module is deprecated and will be removed in a future version.
```

Please update your imports to use the new module paths as described above.

## Backward Compatibility

For now, the old import paths will continue to work but will issue deprecation warnings. These compatibility layers will be removed in a future major version, so it's recommended to update your code as soon as possible.

## Testing Your Changes

After updating your imports, please thoroughly test your application to ensure everything works as expected. Pay special attention to:

1. Error handling and display
2. Configuration loading and saving
3. UI initialization and event handling
4. Any custom error components or handlers

## Getting Help

If you encounter any issues during migration, please:

1. Check the full error message in your logs
2. Verify that all imports are using the new module paths
3. Consult the latest documentation for the new module structure
4. If needed, open an issue with the error details and steps to reproduce

## Timeline

- **Now**: Deprecation warnings are shown when using old import paths
- **Next Major Version**: Old import paths will be removed

Please update your code to use the new import paths before the next major release.

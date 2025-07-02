# Centralized Error Handling and Logging Migration Guide

## Overview

This guide outlines the migration to a new architecture that centralizes error handling and logging, deprecates `logger_bridge`, and enforces a consistent pattern for initializers and handlers across the codebase.

## 1. Base Components

### 1.1 Top-Level Base Handler

All module-specific handlers should inherit from the top-level `BaseHandler`:

```python
# smartcash/ui/handlers/base_handler.py
class BaseHandler:
    """Base class for all UI handlers with common functionality."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_module_logger(self.__class__.__module__)
        self.ui_components = {}
        
    def handle_error(self, error: Exception, context: str = None) -> Dict:
        """Centralized error handling."""
        return handle_error(error, context=context, logger=self.logger)
```

### 1.2 Module-Specific Base Handler

Each module should have its own base handler that inherits from the top-level base handler:

```python
# smartcash/ui/setup/env_config/handlers/base_env_handler.py
from smartcash.ui.handlers.base_handler import BaseHandler

class BaseEnvHandler(BaseHandler):
    """Base handler for environment configuration module."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module_name = "env_config"
        
    def _validate_config(self, config: Dict) -> bool:
        """Common config validation for environment config handlers."""
        # Implementation here
```

## 2. Handler Implementation

### 2.1 Config Handler Example

```python
# smartcash/ui/setup/env_config/handlers/config_handler.py
from .base_env_handler import BaseEnvHandler
from smartcash.ui.handlers.config_handlers import BaseConfigHandler

class EnvConfigHandler(BaseEnvHandler, BaseConfigHandler):
    """Handles environment configuration operations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_config(self) -> Dict:
        """Load configuration with centralized error handling."""
        try:
            # Implementation here
            pass
        except Exception as e:
            return self.handle_error(e, "Failed to load config")
```

## 3. Initializer Implementation

### 3.1 Common Initializer

```python
# smartcash/ui/initializers/common_initializer.py
class CommonInitializer:
    """Base class for all initializers with common functionality."""
    
    def __init__(self, config: Optional[Dict] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_module_logger(self.__class__.__module__)
        self.handlers = {}
        
    def initialize(self) -> Dict:
        """Initialize the module with error handling."""
        try:
            self._pre_initialize()
            result = self._initialize()
            return self._post_initialize(result)
        except Exception as e:
            self.logger.error("Initialization failed", exc_info=True)
            return create_error_response(
                f"Initialization failed: {str(e)}",
                error=e
            )
```

## 4. Migration Steps

### 4.1 Update Dependencies

1. Remove any direct dependencies on `logger_bridge`
2. Add imports for new handler and error utilities

### 4.2 Update Handlers

1. Create module-specific base handler if it doesn't exist
2. Update existing handlers to inherit from appropriate base classes
3. Move common functionality to base handlers
4. Update error handling to use centralized methods

### 4.3 Update Initializers

1. Inherit from `CommonInitializer`
2. Move non-initialization code to appropriate handlers
3. Update error handling to use centralized methods

## 5. Testing

1. Update unit tests to use new handler structure
2. Test error conditions and verify proper error handling
3. Verify logging output is consistent

## 6. Deprecation Notice

- `logger_bridge` is now deprecated
- Direct error handling in initializers is deprecated
- All new code should use the centralized error handling and logging patterns

## 7. Example Migration

### Before:

```python
class OldConfigHandler:
    def __init__(self):
        self.logger = logger_bridge.get_logger(__name__)
        
    def load_config(self):
        try:
            # Implementation
            pass
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {"error": str(e)}
```

### After:

```python
class NewConfigHandler(BaseEnvHandler, BaseConfigHandler):
    def load_config(self):
        try:
            # Implementation
            pass
        except Exception as e:
            return self.handle_error(e, "Failed to load configuration")
```

## 8. Best Practices

1. **Fail Fast**: Validate inputs and fail immediately on error
2. **Single Responsibility**: Keep handlers focused on a single responsibility
3. **Centralized Logging**: Use the provided logging utilities
4. **Consistent Error Handling**: Use the centralized error handling methods
5. **Documentation**: Document all public methods and error conditions

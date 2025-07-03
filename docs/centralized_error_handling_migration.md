# Centralized Error Handling and Logging Migration Guide

## Overview

This guide outlines the migration to a new architecture that centralizes error handling and logging, deprecates `logger_bridge`, and enforces a consistent pattern for initializers and handlers across the codebase.

## 1. Base Components

### 1.1 Top-Level Base Handler

All module-specific handlers should inherit from the top-level `BaseHandler`:

```python
# smartcash/ui/handlers/base_handler.py
from typing import Dict, Any, List, Callable, Literal, TypeVar
import logging
from abc import ABC
from datetime import datetime

from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import (
    handle_ui_errors, 
    safe_execute,
    create_error_response,
    ErrorContext
)
from smartcash.ui.decorators.ui_decorators import (
    safe_ui_operation,
    safe_widget_operation,
    safe_progress_operation,
    safe_component_access
)

class BaseHandler(ABC):
    """Base handler with centralized logging, error handling, and UI utilities.
    
    Features:
    - Centralized logging with consistent module naming
    - Error handling integration with error_handler.py
    - Confirmation dialog utilities
    - Button state management (enable/disable)
    - Status panel update wrapper
    - UI component management helpers
    """
    
    @handle_ui_errors(error_component_title="Handler Initialization Error", log_error=True)
    def __init__(self, module_name: str, parent_module: str = None):
        """Initialize the base handler.
        
        Args:
            module_name: Name of the module
            parent_module: Optional parent module name
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Initialize logger with module-level logging
        self.logger = get_module_logger(f"smartcash.ui.{self.full_module_name}")
        
        # Store confirmation dialog state
        self._confirmation_state = {
            'pending': False,
            'message': '',
            'timestamp': None,
            'timeout_seconds': 120,  # Default 2-minute timeout
            'callback': None
        }
        
        # Log handler for redirecting logs to log_accordion
        self._log_handler = None
        self._log_ui_components = None
```

### 1.2 Module-Specific Base Handler

Each module should have its own base handler that inherits from the top-level base handler. For example, the `BaseDependencyHandler` for the dependency module:

```python
# smartcash/ui/setup/dependency/handlers/base_dependency_handler.py
from typing import Dict, Any, Optional
from smartcash.ui.handlers.base_handler import BaseHandler

class BaseDependencyHandler(BaseHandler):
    """Base handler for dependency module with centralized error handling."""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize base dependency handler with centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.ui_components = ui_components or {}
        self.module_name = "dependency"
```

## 2. Handler Implementation

### 2.1 Config Handler Example

```python
# smartcash/ui/setup/dependency/handlers/config_handler.py
from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler
from .base_dependency_handler import BaseDependencyHandler
from .config_extractor import extract_dependency_config
from .config_updater import update_dependency_ui
from .defaults import get_default_dependency_config

class DependencyConfigHandler(ConfigHandler, BaseDependencyHandler):
    """Config handler for dependency management with centralized error handling"""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize dependency config handler.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config from UI components.
        
        Args:
            ui_components: Dictionary containing UI components
            
        Returns:
            Dictionary containing extracted configuration
        """
        try:
            return extract_dependency_config(ui_components)
        except Exception as e:
            return self.handle_error(e, "Failed to extract dependency configuration")
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components from loaded configuration.
        
        Args:
            ui_components: Dictionary containing UI components
            config: Configuration dictionary to apply
        """
        try:
            update_dependency_ui(ui_components, config)
        except Exception as e:
            self.handle_error(e, "Failed to update UI from configuration")
```

## 3. Inheritance Structure

The centralized error handling system uses a multi-level inheritance structure:

1. **Top-Level Base Handler**: `BaseHandler` provides core error handling, logging, and UI functionality
2. **Config Handler**: `ConfigHandler` inherits from `BaseHandler` and adds configuration management
3. **Module-Specific Base Handler**: `BaseDependencyHandler` inherits from `BaseHandler` and adds module-specific functionality
4. **Concrete Handlers**: Implement specific operations with centralized error handling

### 3.1 Non-Config Handler Inheritance

```
BaseHandler (Top-Level)
    |
    +-- BaseDependencyHandler (Module-Specific)
            |
            +-- OperationHandler
            +-- EventHandler
```

### 3.2 Config Handler Inheritance

```
BaseHandler (Top-Level)
    |
    +-- ConfigHandler (smartcash/ui/handlers/config_handlers.py)
    |       |
    |       +-- [Module-Specific Config Handlers]
    |
    +-- BaseDependencyHandler
            |
            +-- DependencyConfigHandler (Multiple Inheritance)
                    |
                    +-- [Concrete Config Handlers]
```

Note: `DependencyConfigHandler` uses multiple inheritance from both `BaseDependencyHandler` and `ConfigHandler` to combine module-specific functionality with configuration management.

## 5. Base ConfigHandler Implementation

```python
# smartcash/ui/handlers/config_handlers.py
from typing import Dict, Any, Optional, Callable
from abc import abstractmethod
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors

class ConfigHandler(BaseHandler):
    """ConfigHandler with shared configuration management and proper lifecycle handling.
    
    Features:
    - Shared configuration across components using SharedConfigManager
    - Thread-safe operations with proper lifecycle hooks
    - Automatic config change notifications
    - Fallback to local config when shared config is not available
    - Support for non-persistent configuration (in-memory only)
    """
    
    @handle_ui_errors(error_component_title="Config Error", log_error=True)
    def __init__(self, module_name: str, parent_module: str = None, use_shared_config: bool = True,
                 persistence_enabled: bool = True):
        # Initialize BaseHandler first
        super().__init__(module_name, parent_module)
        
        # Config-specific attributes
        self.use_shared_config = use_shared_config
        self.persistence_enabled = persistence_enabled
        self.config_manager = get_config_manager() if persistence_enabled else None
        
        # Initialize shared config manager if enabled and persistence is enabled
        self.shared_manager = None
        if self.persistence_enabled and self.use_shared_config and self.parent_module:
            try:
                self.shared_manager = get_shared_config_manager(self.parent_module)
                # Subscribe to shared config updates
            except Exception as e:
                self.logger.warning(f"Failed to initialize shared config: {e}", exc_info=True)
        
        # Local config state (always available even for non-persistent config)
        self._config_state = ConfigState()
    
    @abstractmethod
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components"""
        pass
        
    @abstractmethod
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI from loaded configuration"""
        pass
    
    @handle_ui_errors(
        error_component_title="Config Load Error",
        log_error=True,
        return_type=dict
    )
    def load_config(self, config_name: Optional[str] = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config with fallback to shared config and base_config.yaml.
        
        For non-persistent handlers, this will always return the default config or
        the current in-memory config state.
        """
        # Implementation details omitted for brevity
        pass
    
    @handle_ui_errors(
        error_component_title="Config Save Error",
        log_error=True,
        return_type=bool
    )
    def save_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None, 
                   update_shared: bool = True) -> bool:
        """Save config with lifecycle hooks and proper error handling.
        
        For non-persistent handlers, this will only update the in-memory state
        and will not attempt to save to disk or shared config.
        """
        # Implementation details omitted for brevity
        pass
```

## 6. Initializer Implementation

### 6.1 Common Initializer

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

## 6. Single Responsibility Principle (SRP) File Organization

To maintain clean code and proper separation of concerns, config handlers should delegate specific functionality to dedicated files:

### 6.1 Required SRP Files for Config Handlers

1. **config_extractor.py**: Contains functions to extract configuration from UI components
   ```python
   # smartcash/ui/module_name/handlers/config_extractor.py
   from typing import Dict, Any
   
   def extract_module_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
       """Extract configuration from UI components.
       
       Args:
           ui_components: Dictionary containing UI components
           
       Returns:
           Dictionary containing extracted configuration
       """
       # Implementation specific to this module
       config = {
           "version": "1.0.0",
           "settings": {}
       }
       
       # Extract values from UI components
       if "setting_input" in ui_components:
           config["settings"]["value"] = ui_components["setting_input"].value
       
       return config
   ```

2. **config_updater.py**: Contains functions to update UI components from loaded configuration
   ```python
   # smartcash/ui/module_name/handlers/config_updater.py
   from typing import Dict, Any
   
   def update_module_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
       """Update UI components from loaded configuration.
       
       Args:
           ui_components: Dictionary containing UI components
           config: Configuration dictionary to apply
       """
       # Implementation specific to this module
       if not config or not isinstance(config, dict):
           return
           
       # Update UI components from config
       if "settings" in config and "value" in config["settings"]:
           if "setting_input" in ui_components:
               ui_components["setting_input"].value = config["settings"]["value"]
   ```

3. **defaults.py**: Contains default configuration values for reset operations
   ```python
   # smartcash/ui/module_name/handlers/defaults.py
   from typing import Dict, Any
   
   # Default configuration for module
   DEFAULT_CONFIG = {
       "version": "1.0.0",
       "created_by": "SmartCash",
       "settings": {
           "value": "default",
           "enabled": True
       }
   }
   
   def get_default_module_config() -> Dict[str, Any]:
       """Get default configuration for module.
       
       Returns:
           Dictionary containing default configuration
       """
       return DEFAULT_CONFIG.copy()
   ```

### 6.2 Benefits of SRP File Organization

1. **Maintainability**: Each file has a clear, focused responsibility
2. **Testability**: Easier to write unit tests for isolated functionality
3. **Readability**: Config handler implementation remains clean and focused on orchestration
4. **Reusability**: Extraction and update logic can be reused in other contexts
5. **Consistency**: Standardized approach across all modules

## 7. Migration Steps

### 7.1 Update Dependencies

1. Remove any direct dependencies on `logger_bridge`
2. Add imports for new handler and error utilities

### 7.2 Update Handlers

1. Create module-specific base handler if it doesn't exist
2. Update existing handlers to inherit from appropriate base classes
3. Move common functionality to base handlers
4. Update error handling to use centralized methods

### 7.3 Update Initializers

1. Inherit from `CommonInitializer`
2. Move non-initialization code to appropriate handlers
3. Update error handling to use centralized methods

## 8. Testing

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

## 8. Shared Functionality from BaseHandler

The `BaseHandler` class provides a rich set of shared functionality that all derived handlers inherit:

### 8.1 Error Handling

- `handle_error(error, context, ui_components, error_title, include_traceback)`: Centralized error handling with consistent logging and UI feedback
- Decorated methods with `@handle_ui_errors`, `@safe_execute`, etc. for automatic error handling

### 8.2 Logging

- Module-level logger automatically configured
- `setup_log_redirection(ui_components)`: Redirects logs to UI components

### 8.3 UI Component Management

- `update_status_panel(ui_components, message, status_type, title)`: Updates status panel with proper error handling
- `clear_ui_outputs(ui_components, output_keys)`: Safely clears UI output components
- `set_buttons_state(ui_components, enabled, button_keys)`: Enables/disables UI buttons
- `disable_all_buttons(ui_components, button_keys)`: Disables all buttons
- `enable_all_buttons(ui_components, button_keys)`: Enables all buttons

### 8.4 Confirmation Dialog Utilities

- `show_confirmation_dialog(ui_components, message, callback, timeout_seconds, title, confirm_text, cancel_text, danger_mode)`: Shows a confirmation dialog with timeout
- `handle_confirmation_result(ui_components, confirmed)`: Handles confirmation dialog result
- `is_confirmation_pending(ui_components)`: Checks if a confirmation dialog is pending

### 8.5 Progress Tracking

- `update_progress(ui_components, value, max_value, message, level)`: Updates progress bar with proper error handling
- `complete_progress(ui_components, message)`: Marks progress as complete
- `error_progress(ui_components, message)`: Marks progress as error
- Multi-level progress tracking: `update_single_progress()`, `update_dual_progress()`, `update_triple_progress()`

## 9. Best Practices

1. **Fail Fast**: Validate inputs and fail immediately on error
2. **Single Responsibility**: Keep handlers focused on a single responsibility
3. **Centralized Logging**: Use the provided logging utilities
4. **Consistent Error Handling**: Use the centralized error handling methods
5. **Documentation**: Document all public methods and error conditions
6. **Safe UI Operations**: Use the safe operation decorators for UI interactions
7. **Proper Inheritance**: Ensure handlers inherit from the appropriate base class
8. **Confirmation Handling**: Use the built-in confirmation dialog utilities for user confirmations

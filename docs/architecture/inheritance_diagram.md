# Core Component Inheritance Structure (Simplified)
```mermaid
classDiagram
    %% Base Classes
    class BaseHandler {
        +logger: UILogger
        +handle_error()
        +validate_inputs()
        +setup_log_suppression()
        +...
    }
    
    class BaseInitializer {
        +initialize()
        +preinitialize()
        +postinitialize()
        +...
    }
    
    %% Handler Inheritance
    class ConfigHandler {
        +load_config(persistance: bool)
        +save_config(persistance: bool)
        +extract_config()
        +update_ui_config()
        +validate_config()
        +...
    }
    
    class UIHandler {
        +update_header()
        +update_status_panel()
        +show_dialog()
        +hide_dialog()
        +update_progress_triple(stages, overall, current)
        +update_dual(overall, current)
        +update_single(current)
        +log_to_ui()
        +handle_events()
        +...
    }
    
    class OperationHandler {
        +execute()
        +execute_operation()
        +stop_operation()
        +validate_operation()
        +handle_result()
    }
    
    %% Initializer Inheritance
    class ModuleInitializer {
        +setup_module()
        +initialize_handlers()
    }
    
    class DisplayInitializer {
        +initialize_display()
    }
    
    class ConfigInitializer {
        +load_configs()
        +setup_config_listeners()
    }
    
    class OperationInitializer {
        +initialize_operations()
        +setup_operation_handlers()
    }
    
    %% Inheritance Relationships
    BaseHandler <|-- ConfigHandler
    BaseHandler <|-- UIHandler
    BaseHandler <|-- OperationHandler
    
    BaseInitializer <|-- ModuleInitializer
    BaseInitializer <|-- DisplayInitializer
    ModuleInitializer <|-- ConfigInitializer
    ModuleInitializer <|-- OperationInitializer
```

## Key Components
This is not fixed, you can expand it

### Handlers
- **BaseHandler**: Core error handling and logging
  - `ConfigHandler`: Manages configuration operations
  - `UIHandler`: Handles UI component management
  - `OperationHandler`: Manages business logic operations

### Initializers
- **BaseInitializer**: Core initialization logic
  - `ModuleInitializer`: Base for module-specific initialization
    - `ConfigInitializer`: Handles configuration initialization
    - `OperationInitializer`: Manages operation setup
  - `DisplayInitializer`: Handles UI display initialization

## Usage
This diagram illustrates the inheritance structure used throughout the SmartCash UI system, ensuring consistent patterns for error handling, configuration management, and UI component initialization.

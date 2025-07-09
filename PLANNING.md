# SmartCash Development Plan

## Overview
Dokumen ini menjelaskan strategi pengembangan SmartCash dengan fokus pada:
1. Menyelesaikan fungsionalitas form UI yang sudah ada
2. Melanjutkan pengembangan modul model/**
3. Mengembangkan modul visualisasi sebagai prioritas terakhir

## Fokus Utama
- **UI Forms**: Memastikan semua form UI yang sudah ada berfungsi dengan baik
- **Modul Model**: Mengembangkan fitur-fitur terkait model machine learning
- **Visualisasi**: Pengembangan visualisasi data dan hasil model (prioritas terakhir)

## Architecture

### Module Groups

#### 1. Setup & Configuration (1.x)
- **1.1 Repository Setup**
  - Module: `setup/repo`
  - Purpose: Handles repository cloning and initialization
  - Persistence Configs: `False`

- **1.2 Google Colab Environment**
  - Module: `setup/colab`
  - Purpose: Manages Google Colab-specific environment configuration
  - Persistence Configs: `False`

- **1.3 Dependency Management**
  - Module: `setup/dependencies`
  - Purpose: Handles package and dependency management
  - Persistence Configs: `True`

#### 2. Data Management (2.x)
- **2.1 Dataset Downloader**
  - Module: `dataset/downloader`
  - Persistence Configs: `True`

- **2.2 Data Splitting**
  - Module: `dataset/split`
  - Persistence Configs: `True`

- **2.3 Data Preprocessing**
  - Module: `dataset/preprocessing`
  - Persistence Configs: `True`

- **2.4 Data Augmentation**
  - Module: `dataset/augmentation`
  - Persistence Configs: `True`

- **2.5 Data Visualization**
  - Module: `dataset/visualization`
  - Persistence Configs: `False`

#### 3. Model Development (3.x)
- **3.1 Pretrained Models**
  - Module: `model/pretrained`
  - Persistence Configs: `True`

- **3.2 Model Architecture**
  - Module: `model/architecture`
  - Persistence Configs: `True`

- **3.3 Model Training**
  - Module: `model/training`
  - Persistence Configs: `True`

- **3.4 Model Evaluation**
  - Module: `model/evaluation`
  - Persistence Configs: `True`

### Core UI Structure

```
smartcash/ui/core/
    ├── __init__.py           # Minimal exports to avoid circular dependencies
    ├── handlers/             # Base handler implementations
    │   ├── __init__.py       # Export only public handler classes
    │   ├── base_handler.py   # Base handler with error handling
    │   ├── config_handler.py
    │   ├── ui_handler.py
    │   └── operation_handler.py
    ├── initializers/         # Initializer implementations
    │   ├── __init__.py       # Export only public initializer classes
    │   ├── base_initializer.py
    │   ├── display_initializer.py
    │   ├── config_initializer.py
    │   ├── module_initializer.py
    │   └── operation_initializer.py
    ├── errors/               # Centralized error handling
    │   ├── __init__.py       # Public error handling API
    │   ├── decorators.py     # Error handling decorators
    │   ├── error_handler.py  # Core error handling
    │   └── exceptions.py     # Custom exceptions
    └── shared/              # Shared utilities
        ├── __init__.py
        ├── logger.py         # Enhanced UILogger
        └── shared_config_manager.py
```

### Key Components

1. **Core Components** (`ui/core/`)
   - **Handlers**: Base handler implementations following a clear hierarchy:
     - `BaseHandler`: Core functionality
     - `ConfigurableHandler`: Config management
       - `PersistentConfigHandler`: File I/O operations
         - `SharedConfigHandler`: Shared configurations
       - `ModuleConfigHandler`: Module-specific configs
   - **Initializers**: Base initialization logic with error handling
   - **Error Handling**: Centralized error management with decorators
   - **Shared Utilities**: Common functionality across modules

2. **Module Structure**
   Each module follows this structure:
   ```
   [module]/
   ├── __init__.py           # Minimal exports, typically just the initializer
   ├── components/           # UI component definitions
   │   ├── __init__.py       # Export only public components
   │   ├── [module]_ui.py    # Main UI components
   │   └── ...
   ├── configs/              # Configuration management
   │   ├── __init__.py
   │   ├── [module]_defaults.py
   │   └── [module]_config_handler.py
   ├── handlers/             # Module-specific handlers
   │   ├── __init__.py
   │   └── [module]_ui_handler.py
   └── operations/           # Operation handlers
   
3. **UI Layout**
   - Consistent container-based layout across modules
   - Standard UI components:
     - Header with title and status
     - Main content area with form components
     - Action buttons
     - Status/logging panel

### Initialization Flow
1. **Environment Setup**
   - Detect and validate environment (Colab/local)
   - Initialize logging and error handling
   - Load shared configurations

2. **Module Initialization**
   - Load module-specific configurations
   - Initialize UI components
   - Set up event handlers
   - Register with core services

3. **Runtime**
   - Handle user interactions
   - Process operations
   - Update UI state
   - Persist configurations

## Scope
- **In Scope**:
  - Refactor initializers in `smartcash/ui/setup`
  - Update handler implementations
  - Ensure container-based layout consistency
  - Maintain existing form layouts and styles
  - Preserve Download, Preprocessing, and Augmentation UIs as-is

- **Out of Scope**:
  - Changes to backend code in `smartcash/dataset` and `smartcash/model`
  - Modifications to UI Core and Components
  - Changes to form layouts and styles
  - Modifications to Download, Preprocessing, and Augmentation UIs

## Implementation Strategy
1. **Phase 1: Base Infrastructure**
   - Create base initializer and handler interfaces
   - Implement container management system
   - Set up centralized logging and error handling

2. **Phase 2: Module Initializers**
   - Refactor existing initializers one module at a time
   - Implement new handler patterns
   - Add container-based layout support

3. **Phase 3: Integration**
   - Update module initialization flow
   - Ensure backward compatibility
   - Add comprehensive tests

## Success Metrics
- 100% test coverage for new code
- 100% UI test
- 100% Cells execution test
- No regression in existing functionality
- Consistent container-based layout
- Improved code maintainability

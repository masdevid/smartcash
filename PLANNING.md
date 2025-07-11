# SmartCash Development Plan

## Overview
SmartCash UI system has achieved **81.9% overall success rate** with solid core infrastructure. This document outlines the next development priorities based on comprehensive testing results.

## Architecture
- **New Clean Architecture Pattern**
 - **Class Diagram Example** look at `inheritance_diagram.md`
  ```
  BaseHandler → Error Handling → SharedConfig → UI Components
       ↓              ↓              ↓              ↓
    Lightweight    Focused       Centralized   Specialized
    Fast init      Clear         Thread-safe   Component-specific
    Simple API     Reliable      Versioning    Action-based
  ```
### Module Groups

#### 1. Setup & Configuration (1.x)
- **1.1 Repository Setup**
  - Module: `setup/repo` (direct cell code)
  - Purpose: Handles repository cloning and initialization
  - Persistence Configs: `False`

- **1.2 Google Colab Environment**
  - Module: `setup/colab`
  - Purpose: Manages Google Colab-specific environment configuration
  - Persistence Configs: `False`
  - Status: 100% functional, full UI standardization compliance
  - Test Coverage: 100% (14/14 tests passing)

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
    │   ├── base_handler.py         # Base handler with error handling + UILogger (with Suppression)
    │   ├── config_handler.py       # Config Handler (extend UI Handler)
    │   ├── ui_handler.py           # UI Handler (extend Base Handler)
    │   └── operation_handler.py    # Operation Handler (extend UI Handler)
    ├── initializers/         # Initializer implementations
    │   ├── __init__.py       # Export only public initializer classes
    │   ├── base_initializer.py      # Base Initializer
    │   ├── display_initializer.py   # Display Initializer (extend Base Initializer)
    │   ├── config_initializer.py    # Config Initializer (extend Module Initializer)
    │   ├── module_initializer.py    # Module Initializer (extend Base Initializer)
    │   └── operation_initializer.py # Operation Initializer (extend Module Initializer)
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
     - `BaseHandler`: Core functionality (extend Error Handler)
     - `ConfigurableHandler`: Config management
       - `PersistentConfigHandler`: File I/O operations
         - `SharedConfigHandler`: Shared configurations
       - `ModuleConfigHandler`: Module-specific configs
   - **Initializers**: Base initialization logic centralized with error handling
   - **Error Handling**: Centralized error management with decorators
   - **Shared Utilities**: Common functionality across modules

2. **Module Structure**
   Each module follows this structure:
   ```
   [module]/
   ├── __init__.py               # Minimal exports, typically just the initializer
   ├── constants.py              # Constants management
   ├── [module]_initializer.py   # Main Initializer + Display Initializer
   ├── components/               # UI component definitions
   │   ├── __init__.py           # Export only public components
   │   ├── [module]_ui.py        # Main UI components
   │   └── ...
   ├── configs/                  # Configuration management
   │   ├── __init__.py           # Export only public configs
   │   ├── [module]_defaults.py  # Default Config
   │   └── [module]_config.py    # Config Handler
   ├── handlers/                 # Module-specific handlers
   │   ├── __init__.py
   │   └── [module]_ui_handler.py # UI Handler
   ├── operations/                # Operation handlers
   │   ├── __init__.py
   │   ├── [module]_manager.py   # Operation Manager (Orchestrator -> Inherit Operation Handler)
   │   └── [operation_name]_operation.py # Operation Handler (Inherit Manager)
   └── services/                 # Module-specific services (Backend Bridge Layer)
   
3. **UI Layout**
   - Consistent container-based layout across modules
   - Read more in `UI_MODULE_STANDARDIZATION_SUMMARY.md`

### Initialization Flow
1. **Environment Setup**
   - Detect and validate environment (Colab/local)
   - Initialize logging and error handling
   - Load shared configurations

## Testing Strategy

### Test Organization

1. **Test Directory Structure**
   - Semua test harus diorganisir dalam folder `tests/` dengan struktur yang mencerminkan struktur kode utama
   - Gunakan pola mirror untuk test files (contoh: `smartcash/ui/model/backbone.py` -> `tests/ui/model/test_backbone.py`)
   - Kelompokkan test terkait dalam folder yang sesuai (unit, integration, e2e)

2. **Test Naming Convention**
   - Nama file test harus diawali dengan `test_`
   - Nama fungsi test harus deskriptif dan menjelaskan behavior yang di-test
   - Gunakan format: `test_<method>_<scenario>_<expected_behavior>`

3. **Comprehensive Test Suite**
   - Gunakan `tests/test_all_module_comprehensive.py` untuk menjalankan seluruh test suite
   - Pastikan test suite dapat dijalankan secara terpisah untuk setiap modul
   - Gunakan marker atau kategori untuk mengelompokkan test yang terkait

4. **Test Dependencies**
   - Gunakan `conftest.py` untuk shared test fixtures
   - Pisahkan test data dari kode test
   - Gunakan factory pattern untuk membuat test data yang kompleks

5. **Test Coverage**
   - Targetkan minimal 80% code coverage untuk kode inti
   - Prioritaskan test untuk UI, critical paths dan business logic

6. **Test Documentation**
   - Dokumentasikan asumsi dan skenario test yang kompleks
   - Gunakan docstring untuk menjelaskan tujuan test
   - Sertakan contoh input/output yang diharapkan untuk test yang kompleks

7. **Test Maintenance**
   - Update test ketika ada perubahan requirements
   - Refactor test yang redundan
   - Review test secara berkala untuk memastikan relevansi

8. **Module Initialization**
   - Load module-specific configurations
   - Initialize UI components
   - Set up event handlers
   - Register with core services

9. **Runtime**
   - Handle user interactions
   - Process operations
   - Update UI state
   - Persist configurations

## Scope
- **In Scope**:
  - Refactor `smartcash/ui/core` dan `smartcash/ui/components`
  - Refactor initializers in `smartcash/ui/{setup, dataset, model}`
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

### **Phase 4: Dataset Pipeline Enhancement (Next Quarter)**
**Priority: HIGH** - Complete the data processing workflow

1. **Data Preprocessing Module** (Current: 75% → Target: 95%)
   - Fix remaining import issues in test suite
   - Enhance error handling for edge cases
   - Fix Progress Tracker
   - Fix Confirmation Dialog
   - Fix Collapsible Logger 

2. **Data Splitting Module** (Current: 78.8% → Target: 90%)
   - Fix UI and Config Handler
   - Resolve failing test cases (21/99 tests failing)

3. **Data Augmentation Module** (Current: 85% → Target: 95%)
   - Fix remaining import/initialization issues
   - Enhance error handling for edge cases
   - Fix Progress Tracker
   - Fix Confirmation Dialog
   - Fix Collapsible Logger 

### **Phase 5: Training Module Stabilization (Following Quarter)**
**Priority: HIGH** - Achieve production-ready training pipeline

1. **Training Service Stability** (Current: 79.3% → Target: 95%)
   - Fix simulation training issues (9/11 tests passing)
   - Improve backend integration (1/3 tests passing)
   - Complete training pipeline workflow (0/2 tests passing)
   - Enhance error recovery and resume capabilities

2. **Training UI Enhancements**
   - Improve real-time monitoring
   - Add better progress visualization
   - Enhance hyperparameter tuning interface
   - Add training history and comparison tools

### **Phase 6: Data Visualization Implementation (Next 6 Months)**
**Priority: MEDIUM** - Build comprehensive visualization suite

1. **Core Visualization Infrastructure**
   - Design visualization framework
   - Implement base chart components
   - Create interactive plotting capabilities
   - Add export functionality

2. **Dataset Visualizations**
   - Data distribution plots
   - Class balance visualization
   - Augmentation previews
   - Data quality metrics

3. **Model Visualizations**
   - Training progress charts
   - Model performance metrics
   - Confusion matrices
   - Feature importance plots

## Success Metrics
### **Immediate Goals (Q1 2025)**
- UI Module Standardization: 100% success rate
- Core Module Refactoring: 90%+ success rate across all modules
- Shared UI Components:100% success rate across all modules
- Setup Cells Module: 100% success rate across all modules
- Dataset Pipeline: 90%+ success rate across all modules
- Training Module: 95%+ success rate
- Overall System: 90%+ success rate
- Zero critical runtime errors

### **Medium-term Goals (Q2 2025)**
- Data Visualization: 85%+ success rate
- Advanced training features: 90%+ success rate
- User workflow completion rate: 95%+
- Performance benchmarks met

### **Long-term Goals (2025)**
- 95%+ success rate across all modules
- Complete feature parity with requirements
- Production deployment ready
- Comprehensive documentation and user guides

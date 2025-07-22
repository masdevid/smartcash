# UI Module Structure

## Overview

This document outlines the standard structure for UI modules in the SmartCash application. Each module follows a consistent organization pattern that promotes maintainability and separation of concerns.

## Module Group Mapping

Based on the cells structure, here's how modules are organized into functional groups:

### 1. Setup & Configuration (1.x)
- **1.1 Repository Setup** (`cell_1_1_repo_clone.py`)
  - Module: `setup/repo`
  - Purpose: Handles repository cloning and initialization
  - Persistence Configs: `False`

- **1.2 Google Colab Environment** (`cell_1_2_colab.py`)
  - Module: `setup/colab`
  - Purpose: Manages Google Colab-specific environment configuration and setup
  - Persistence Configs: `False`

- **1.3 Dependency Management** (`cell_1_3_dependency.py`)
  - Module: `setup/dependencies`
  - Purpose: Handles package and dependency management
  - Persistence Configs: `True`

### 2. Data Management (2.x)
- **2.1 Dataset Downloader** (`cell_2_1_downloader.py`)
  - Module: `dataset/downloader`
  - Purpose: Manages dataset downloading and extraction
  - Persistence Configs: `True`

- **2.2 Data Splitting** (`cell_2_2_split.py`)
  - Module: `dataset/split`
  - Purpose: Handles train/validation/test splitting
  - Persistence Configs: `True`

- **2.3 Data Preprocessing** (`cell_2_3_preprocess.py`)
  - Module: `dataset/preprocessing`
  - Purpose: Implements data preprocessing pipelines
  - Persistence Configs: `True`

- **2.4 Data Augmentation** (`cell_2_4_augment.py`)
  - Module: `dataset/augmentation`
  - Purpose: Manages data augmentation operations
  - Persistence Configs: `True`

- **2.5 Data Visualization** (`cell_2_5_visualize.py`)
  - Module: `dataset/visualization`
  - Purpose: Handles data visualization and exploration
  - Persistence Configs: `False`

### 3. Model Development (3.x)
- **3.1 Pretrained Models** (`cell_3_1_pretrained.py`)
  - Module: `model/pretrained`
  - Purpose: Manages pretrained model loading and setup
  - Persistence Configs: `True`

- **3.2 Model Architecture** (`cell_3_2_backbone.py`)
  - Module: `model/architecture`
  - Purpose: Defines model architectures and backbones
  - Persistence Configs: `True`

- **3.3 Model Training** (`cell_3_3_train.py`)
  - Module: `model/training`
  - Purpose: Handles model training workflows
  - Persistence Configs: `True`

- **3.4 Model Evaluation** (`cell_3_4_evaluate.py`)
  - Module: `model/evaluation`
  - Purpose: Manages model evaluation and metrics based on research objectives and limitations
  - Persistence Configs: `True`

## Standard Module Structure

```
smartcash/ui/[group]/[module]/
    ├── __init__.py           # Minimal exports, typically just the initializer
    ├── components/           # UI component definitions
    │   ├── __init__.py       # Export only public components
    │   ├── [module]_ui.py
    │   ├── buttons.py
    │   ├── forms.py
    │   └── panels.py
    ├── configs/              # Configuration management (SRP)
    │   ├── __init__.py       # Export only get_config_handler
    │   ├── [module]_defaults.py      # Default minimal config + YAML definition
    │   ├── [module]_extractor.py     # UI config extraction logic (if functions too big)
    │   ├── [module]_updater.py       # UI update from config logic (if functions too big)
    │   ├── [module]_validator.py     # Config validation logic (if functions too big)
    │   └── [module]_config_handler.py       # Config handler implementation
    ├── handlers/            # Module-specific handlers
    │   ├── __init__.py      # Export only public handlers
    │   ├── [module]_ui_handler.py
    │   └── [specific]_handler.py
    ├── operations/          # Operation handlers
    │   ├── __init__.py      # Export only public operation handlers
    │   ├── [operation1]_operation.py
    │   └── [operation2]_operation.py
    ├── services/            # Backend bridge services (Optional)
    │   ├── __init__.py      # Export only public services
    │   ├── [module]_service.py
    │   └── [specific]_service.py
    ├── constants.py         # Module-specific constants
    └── [module]_initializer.py  # Module initializer
```
## Files & Directory Responsibilities
- `__init__.py`: Export only public components
- `constants.py`: Define constants specific to this module
- `[module]_initializer.py`: Initialize module, inherit from `core/initializers/module_initializer.py`

### components/
- Contains UI component definitions
- Each component should be self-contained
- Exports only public components through `__init__.py`
- Follows the principle of composition over inheritance
- Using defaults container based component with order:
    - Header Container:
        - Header (must)
        - Status Panel (must)
    - Form Container (Custom to each module)
    - Action Container:
        - Save/Reset Buttons (only if need persistence config) | (float right)
        - Big Primary Buttons (for single operation only) | (float center)
        - Action Buttons (for multiple operations) | (float left)
        - Big Primary Buttons can't reside with Action Buttons.
        - Don't use Primary Buttons if there are multiple main operations by click (save_reset operation excluded)
    - Summary Container (Custom to each module) (Nice to have)
    - Operation Container:
        - Progress Tracker (must)
        - Dialog Confirmation Area (Opsional)
        - Log Accordion (must)
    - Footer Container:
        - Info Accordion(s) (Nice to have)
        - Tips Panel (opsional)
- `[module]_ui.py`: Contains main UI component definitions
- `*_section.py`: Contains section UI component definitions
- `*_panel.py`: Contains panel UI component
- `*_widget.py`: Contains specific widget UI component
- `*[what]_info.py`: Contains collapsible infos or guidances, but must be placed in dedicated `info_box/` folder

### configs/
- Manages all configuration-related functionality. 
- Split codes into multiple files is a must if `[module]_config_handler.py` has more than 500 lines
- `[module]_defaults.py`: Defines default configuration values
- `[module]_extractor.py`: Extracts configuration from UI components if too complex
- `[module]_updater.py`: Updates UI components from configuration if too complex
- `[module]_validator.py`: Validates configuration values if too complex
- `[module]_config_handler.py`: Implements configuration handling logic (Inherit `core/handlers/config_handler.py`)

### handlers/
- Contains module-specific handler implementations
- Each handler should have a single responsibility
- Should inherit from appropriate base handlers. For main handler, inherit from `core/handlers/ui_handler.py`
- Handles user interactions and business logic
- naming: `[module]_ui_handler.py` (for main handler), `[module]_specific_handler.py` (for specific handler)
- `[module]_ui_handler.py` should inherit `smartcash/ui/core/handlers/ui_handler.py`

### operations/ 
- Contains operation-specific handlers
- Implements complex operations that may span multiple components
- Handles long-running tasks with progress reporting
- Manages operation state and cleanup
- naming: `[name]_operation.py`
- `[name]_operation.py` should inherit `smartcash/ui/core/handlers/operation_handler.py`

### services/
- Optional if no backend functionality is needed
- Contains service classes/helpers
- It's not UI utility/helper folders. Should leverage parent class shared functions instead of creating utility/helper files
- Acts as a bridge to backend functionality such as progress and log callbacks
- Implements API calls and data transformations
- Handles error conditions and retries
- Provides a clean interface for UI components
- naming: `[name]_service.py`

## Best Practices

1. **Minimal Exports**: Only expose necessary components through `__init__.py`
2. **Single Responsibility**: Each file and class should have a single responsibility
3. **Dependency Injection**: Pass dependencies explicitly rather than importing them
4. **Type Hints**: Use type hints for better IDE support and code clarity
5. **Documentation**: Document public APIs with docstrings
6. **Testing**: Include unit tests for all non-trivial logic
7. **Error Handling**: Handle errors gracefully and provide meaningful messages
8. **Configuration**: Keep configuration separate from code
9. **Constants**: Define constants in `constants.py`
10. **Naming**: Use consistent naming conventions across the module

This structure ensures consistency across all UI modules in the SmartCash application, making the codebase more maintainable and easier to understand for new developers.

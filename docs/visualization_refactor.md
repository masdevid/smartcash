# Visualization Package Refactor

## Overview

This document describes the refactoring of the visualization components from a single `visualization_manager.py` file into a modular package structure. The new structure improves maintainability, testability, and extensibility of the visualization code.

## New Package Structure

```
smartcash/model/training/visualization/
├── __init__.py           # Package initialization and public API
├── manager.py            # Main VisualizationManager class
├── types.py              # Type hints and data classes
├── charts/               # Chart generation modules
│   ├── __init__.py
│   ├── base.py           # Base chart class and utilities
│   ├── training.py       # Training-specific charts
│   ├── confusion.py      # Confusion matrix charts
│   ├── metrics.py        # Metrics comparison charts
│   └── research.py       # Research dashboard charts
└── utils/                # Utility functions
    ├── __init__.py
    └── validators.py     # Input validation utilities
```

## Key Changes

1. **Modular Design**:
   - Split the monolithic `visualization_manager.py` into focused modules
   - Each chart type has its own module with clear responsibilities
   - Common utilities and types are separated for better organization

2. **Improved Type Safety**:
   - Added comprehensive type hints throughout the codebase
   - Created data classes for configuration and state management
   - Used enums for fixed sets of values (e.g., `ChartType`, `MetricType`)

3. **Better Error Handling**:
   - Added input validation for all public methods
   - Improved error messages and logging
   - Graceful degradation when visualization dependencies are missing

4. **Enhanced Testability**:
   - Added comprehensive unit tests for all components
   - Mocked external dependencies for reliable testing
   - Test coverage for error conditions and edge cases

## Migration Guide

### Old Code

```python
from smartcash.model.training.visualization_manager import VisualizationManager

# Initialize
viz_manager = VisualizationManager(num_classes_per_layer={"banknote": 2, "denomination": 5})

# Update metrics
viz_manager.update_with_research_metrics(
    epoch=1,
    phase="train",
    metrics={"loss": 0.5, "accuracy": 0.9},
    learning_rate=0.001
)

# Generate charts
viz_manager.generate_charts()
```

### New Code

```python
from smartcash.model.training.visualization import VisualizationManager

# Initialize
viz_manager = VisualizationManager(
    num_classes_per_layer={"banknote": 2, "denomination": 5},
    save_dir="visualizations",
    verbose=True
)

# Update metrics
viz_manager.update_metrics(
    epoch=1,
    phase="train",
    metrics={"loss": 0.5, "accuracy": 0.9},
    learning_rate=0.001
)

# Generate all charts
viz_manager.generate_all_charts()

# Or generate specific charts
training_curve_path = viz_manager.generate_training_curves()
confusion_matrix_paths = viz_manager.generate_confusion_matrices()
```

## New Features

1. **Selective Chart Generation**:
   - Generate specific charts instead of all at once
   - More control over chart appearance and behavior

2. **Improved Configuration**:
   - Fine-grained control over chart appearance
   - Support for custom styling and theming
   - Configurable output formats and resolutions

3. **Resource Management**:
   - Explicit cleanup of resources with `cleanup()` method
   - Automatic cleanup in `__del__`
   - Better memory management for large visualizations

## Deprecation Notice

The old `visualization_manager.py` has been moved to `archive/visualization_refactor/` for reference. It will be removed in a future release. Please migrate to the new visualization package as soon as possible.

## Testing

Run the visualization tests with:

```bash
pytest tests/unit/test_visualization.py -v
```

Or run all tests with:

```bash
pytest
```

## Contributing

When adding new chart types or features:

1. Create a new module in the `charts` directory
2. Extend the `BaseChart` class
3. Add comprehensive tests
4. Update this documentation

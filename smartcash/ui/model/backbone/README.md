# Backbone Model UI Module

Module untuk konfigurasi backbone model YOLOv5 dengan EfficientNet-B4.

## Structure

```
backbone/
├── __init__.py
├── backbone_init.py         # Main initializer extending ConfigCellInitializer
├── components/              # UI components
│   ├── __init__.py
│   ├── ui_components.py    # Main UI assembly
│   ├── model_form.py       # Model configuration form
│   └── config_summary.py   # Configuration summary display
├── handlers/               # Event handlers
│   ├── __init__.py
│   ├── model_handler.py    # Main model operations handler
│   ├── config_handler.py   # Configuration management
│   └── api_handler.py      # API integration handler
└── utils/                  # Utilities
    ├── __init__.py
    ├── ui_utils.py         # UI helper functions
    ├── config_utils.py     # Config extraction/update
    └── validation.py       # Configuration validation
```

## Usage

```python
from smartcash.ui.model.backbone import BackboneInitializer

# Initialize
initializer = BackboneInitializer()
container = initializer.initialize()
display(container)
```

## Features

- EfficientNet-B4 and CSPDarknet backbone selection
- Single/Multi-layer detection modes
- Feature optimization controls
- Mixed precision training support
- Auto-save configuration to model_config.yaml
- Integration with shared configuration manager
- Real-time progress tracking
- Model validation and info display

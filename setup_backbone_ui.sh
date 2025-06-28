#!/bin/bash

# File: setup_backbone_ui.sh
# Deskripsi: Script untuk membuat struktur folder dan file untuk Backbone Model UI

echo "🚀 Setting up Backbone Model UI structure..."

# Base directory
BASE_DIR="smartcash/ui/model/backbone"

# Create directories
echo "📁 Creating directories..."
mkdir -p "$BASE_DIR/components"
mkdir -p "$BASE_DIR/handlers"
mkdir -p "$BASE_DIR/utils"

# Create __init__.py files
echo "📄 Creating __init__.py files..."
touch "$BASE_DIR/__init__.py"
touch "$BASE_DIR/components/__init__.py"
touch "$BASE_DIR/handlers/__init__.py"
touch "$BASE_DIR/utils/__init__.py"

# Create component files
echo "📄 Creating component files..."
touch "$BASE_DIR/components/ui_components.py"
touch "$BASE_DIR/components/model_form.py"
touch "$BASE_DIR/components/config_summary.py"

# Create handler files
echo "📄 Creating handler files..."
touch "$BASE_DIR/handlers/model_handler.py"
touch "$BASE_DIR/handlers/config_handler.py"
touch "$BASE_DIR/handlers/api_handler.py"

# Create util files
echo "📄 Creating util files..."
touch "$BASE_DIR/utils/ui_utils.py"
touch "$BASE_DIR/utils/config_utils.py"
touch "$BASE_DIR/utils/validation.py"

# Create main initializer
echo "📄 Creating main initializer..."
touch "$BASE_DIR/backbone_init.py"

# Create README
echo "📄 Creating README..."
cat > "$BASE_DIR/README.md" << 'EOF'
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
EOF

# Set permissions
chmod +x "$BASE_DIR/backbone_init.py"

echo "✅ Backbone Model UI structure created successfully!"
echo ""
echo "📁 Structure created:"
tree "$BASE_DIR" 2>/dev/null || find "$BASE_DIR" -type f | sort

echo ""
echo "📝 Next steps:"
echo "1. Copy the artifact code to respective files"
echo "2. Update smartcash/ui/model/__init__.py to include BackboneInitializer"
echo "3. Test the UI module in Jupyter notebook"
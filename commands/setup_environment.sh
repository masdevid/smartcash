#!/bin/bash

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install ipython ipykernel ipywidgets

# Install project in development mode
echo "Installing project in development mode..."
pip install -e .

echo "\n✅ Environment setup complete!"
echo "To activate the virtual environment, run:"
echo "source .venv/bin/activate"
echo "\nTo verify the setup, run:"
echo "python -c 'from smartcash.ui.setup.env_config.components.ui_factory import UIFactory; print(\"✅ UIFactory imported successfully!\")'"

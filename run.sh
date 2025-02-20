#!/bin/bash

# Set strict error handling
set -e

# Function to check if we're in the correct directory
check_directory() {
    if [ ! -f "setup.py" ]; then
        echo "Error: Please run this script from the smartcash project root directory"
        exit 1
    fi
}

# Function to setup conda environment
setup_conda() {
    if command -v conda &> /dev/null; then
        # Add conda-forge to avoid the defaults warning
        conda config --add channels conda-forge
        conda config --set channel_priority strict
        
        # Check if environment exists
        if conda env list | grep -q "smartcash"; then
            echo "Activating smartcash environment..."
            eval "$(conda shell.bash hook)"
            conda activate smartcash
        else
            echo "Creating smartcash environment..."
            conda create -n smartcash python=3.9 -y
            eval "$(conda shell.bash hook)"
            conda activate smartcash
            pip install -e .
        fi
    else
        echo "Warning: conda not found, proceeding without environment activation"
    fi
}

# Main execution
check_directory
setup_conda

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Run the program
echo "Starting SmartCash..."
python run.py

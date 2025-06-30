#!/bin/bash

# Exit on error
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install conda first."
    exit 1
fi

# Check if smartcash_test environment exists
if ! conda env list | grep -q "smartcash_test"; then
    echo "Creating conda environment smartcash_test..."
    conda create -n smartcash_test python=3.9 -y
    echo "Environment created."
else
    echo "Using existing conda environment: smartcash_test"
fi

# Activate the environment and install dependencies
echo "Activating environment and installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate smartcash_test

# Install requirements
pip install -r tests/requirements-test.txt

# Install package in development mode
pip install -e .

# Run the tests
echo "Running integration tests..."
cd tests/integration
pytest test_progress_tracking.py -v --html=test_report.html

# Open the test report in default browser
if command -v xdg-open &> /dev/null; then
    xdg-open test_report.html
elif command -v open &> /dev/null; then
    open test_report.html
fi

echo "Test execution complete. Report generated: tests/integration/test_report.html"

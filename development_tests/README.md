# Development Test Scripts

This directory contains test scripts that were used during development and debugging of various modules. These are preserved for reference but are not part of the main test suite.

## Files:

### Cell Execution Tests
- `test_all_cells.py` - Tests execution of all three main cell files
- `test_execution_colab.py` - Tests colab module UI initialization 
- `test_execution_dependency.py` - Tests dependency module execution
- `test_execution_downloader.py` - Tests downloader module execution
- `test_colab_runner.py` - Colab module runner and validator

### Module-Specific Tests
- `test_backup_functionality.py` - Tests backup/restore functionality for pretrained module
- `test_model_builder_integration.py` - Tests model builder integration
- `test_pretrained_ui.py` - Tests pretrained module UI functionality

## Usage

These scripts are standalone and can be run individually for debugging or development purposes. They are not part of the main test suite which is located in the `tests/` directory.

## Note

The organized test suite in `tests/` directory should be used for regular testing and CI/CD. These development scripts are kept for reference and debugging purposes only.
# Test Structure

This directory contains the test suite for the SmartCash project. The test structure mirrors the main project structure for better organization and maintainability.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py
├── test_imports.py
├── common/                # Tests for common utilities
│   └── __init__.py
├── ui/                    # Tests for UI components
│   ├── __init__.py
│   ├── components/        # Tests for UI components
│   │   ├── __init__.py
│   │   ├── test_footer_container.py
│   │   └── ...
│   └── setup/            # Tests for UI setup
│       ├── __init__.py
│       └── env_config/
│           └── handlers/
│               └── test_folder_handler.py
└── dataset/              # Tests for dataset functionality
    └── __init__.py
```

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/ui/components/test_footer_container.py -v
```

To run tests with coverage:

```bash
pytest --cov=smartcash --cov-report=term-missing
```

## Writing Tests

1. Test files should be named `test_*.py`
2. Test classes should start with `Test`
3. Test methods should start with `test_`
4. Use fixtures for common test setup/teardown
5. Keep tests focused and independent
6. Mock external dependencies when necessary

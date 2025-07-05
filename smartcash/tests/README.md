# Test Structure

This directory contains the test suite for the SmartCash project. The test structure follows Python best practices with a clear separation between unit and integration tests.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py                # Global pytest configuration and fixtures
├── test_helpers.py            # Shared test utilities and helpers
├── unit/                      # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── core/                  # Tests for core functionality
│   │   ├── __init__.py
│   │   ├── test_core_imports.py
│   │   └── ...
│   └── ui/                    # Tests for UI components
│       ├── __init__.py
│       ├── components/        # Tests for UI components
│       │   ├── __init__.py
│       │   ├── test_header_container.py
│       │   ├── test_footer_container.py
│       │   └── ...
│       └── setup/             # Tests for UI setup
│           ├── __init__.py
│           └── dependency/    # Tests for dependency management
│               ├── __init__.py
│               ├── test_dependency_error_handling.py
│               ├── test_dependency_initializer_isolated.py
│               └── test_dependency_initializer_simple.py
└── integration/              # Integration tests (slower, test interactions)
    └── __init__.py
```

## Running Tests

### Running All Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=smartcash --cov-report=term-missing

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Running Specific Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run a specific test file
pytest tests/unit/ui/components/test_header_container.py -v

# Run tests matching a pattern
pytest -k "test_header" -v

# Run only failed tests
pytest --last-failed
```

### Test Coverage

```bash
# Generate HTML coverage report
pytest --cov=smartcash --cov-report=html

# Open the HTML report (on macOS)
open htmlcov/index.html
```

## Writing Tests

### Test Organization

- **Unit Tests**: Fast, isolated tests that verify individual components
- **Integration Tests**: Test interactions between components
- **Fixtures**: Defined in `conftest.py` or local `__init__.py` files
- **Helpers**: Shared test utilities in `test_helpers.py`

### Best Practices

1. **Test Naming**:
   - Use `test_` prefix for test functions
   - Name tests to describe the behavior being tested
   - Example: `test_header_container_initialization()`

2. **Test Structure**:
   ```python
   def test_feature_behavior():
       # Arrange - setup test data and objects
       obj = SomeClass()
       
       # Act - perform the action being tested
       result = obj.method()
       
       # Assert - verify the outcome
       assert result == expected_value
   ```

3. **Fixtures**:
   - Use `@pytest.fixture` for reusable test components
   - Keep fixtures close to where they're used
   - Use `autouse=True` for fixtures that should run for every test

4. **Mocks**:
   - Use `unittest.mock` or `pytest-mock` for isolating tests
   - Prefer dependency injection over monkey patching

5. **Assertions**:
   - Use descriptive assertion messages
   - Leverage pytest's assertion rewriting
   - Use helper functions from `test_helpers.py` for common assertions

### Running Tests in Development

```bash
# Run tests continuously on file changes (requires pytest-watch)
ptw -- tests/unit/

# Run tests with detailed output
pytest -v --tb=short

# Run tests and stop after first failure
pytest -x

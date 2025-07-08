"""
Pytest configuration and plugins for the test suite.
"""
import os
import sys
from pathlib import Path
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def pytest_collection_modifyitems(config, items):
    """Prevent collection of test files outside the tests directory."""
    tests_dir = Path(project_root) / "tests"
    
    for item in items[:]:  # Iterate over a copy of the list
        test_path = Path(item.fspath).resolve()
        
        # Skip if the test is in the tests directory
        if tests_dir in test_path.parents:
            continue
            
        # Remove the test from collection
        items.remove(item)
        
        # Print a warning
        config.warn(
            code="C1",
            message=f"Test file not in tests directory: {test_path.relative_to(project_root)}"
        )

# Add marker to track test locations
# Usage: @pytest.mark.integration or @pytest.mark.unit
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test (slower, tests interactions)"
    )

# Add a command-line option to run only unit or integration tests
def pytest_addoption(parser):
    parser.addoption(
        "--unit",
        action="store_true",
        default=False,
        help="Run only unit tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run only integration tests"
    )

def pytest_runtest_setup(item):
    """Skip tests based on command-line options."""
    if item.config.getoption("--unit"):
        if "integration" in item.keywords:
            pytest.skip("Skipping integration test (--unit given)")
    elif item.config.getoption("--integration"):
        if "unit" in item.keywords:
            pytest.skip("Skipping unit test (--integration given)")

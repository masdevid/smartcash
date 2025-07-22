"""
File: tests/test_helpers/fixture_helpers.py
Description: Pytest fixtures and fixture-related helpers.
"""
import pytest
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from unittest.mock import MagicMock, patch, create_autospec

from ..test_helpers.mock_objects import create_mock_config, create_mock_ui_components

T = TypeVar('T')

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Fixture providing a default mock configuration."""
    return create_mock_config()


@pytest.fixture
def mock_ui_components() -> Dict[str, Any]:
    """Fixture providing mock UI components."""
    return create_mock_ui_components()


@pytest.fixture
def mock_operation_handler():
    """Fixture providing a mock operation handler."""
    from ..test_helpers.mock_objects import MockOperationHandler
    return MockOperationHandler()


def patch_class(
    target: str,
    autospec: bool = True,
    **kwargs
) -> Callable:
    """Helper to patch a class with autospec by default.
    
    Args:
        target: The target class to patch (as a string)
        autospec: Whether to use autospec (default: True)
        **kwargs: Additional arguments to pass to patch
        
    Returns:
        A decorator that applies the patch
    """
    return patch(target, autospec=autospec, **kwargs)


def patch_module_imports(module_paths: Dict[str, Any]) -> Callable:
    """Helper to patch multiple module imports at once.
    
    Args:
        module_paths: Dictionary mapping module paths to mock objects or values
        
    Returns:
        A context manager that patches the specified imports
    """
    patches = [
        patch(path, new=value) 
        for path, value in module_paths.items()
    ]
    
    class PatchContext:
        def __enter__(self):
            for p in patches:
                p.start()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            for p in patches:
                p.stop()
    
    return PatchContext()


def create_fixture(
    fixture_func: Callable[..., T],
    scope: str = "function",
    autouse: bool = False
) -> Callable[..., T]:
    """Helper to create a fixture with common settings.
    
    Args:
        fixture_func: The fixture function
        scope: The scope of the fixture (default: "function")
        autouse: Whether the fixture should be used automatically (default: False)
        
    Returns:
        A pytest fixture
    """
    return pytest.fixture(scope=scope, autouse=autouse)(fixture_func)

"""
File: tests/test_helpers/mock_objects.py
Description: Mock objects and factories for testing.
"""
from typing import Dict, Any, Optional, List, Type, TypeVar, Callable
from unittest.mock import MagicMock, create_autospec, PropertyMock

class MockOperationHandler:
    """Base mock operation handler for testing UI operations."""
    
    def __init__(self, *args, **kwargs):
        self.ui_module = kwargs.get('ui_module')
        self.config = kwargs.get('config', {})
        self.callbacks = kwargs.get('callbacks', {})
        self.init_args = args
        self.init_kwargs = kwargs
        self._execute_return = {'success': True, 'message': 'Success'}
        self.execute = MagicMock(return_value=self._execute_return)
        
        if hasattr(self, 'should_fail') and self.should_fail:
            self.execute.side_effect = Exception("Test error")


def create_mock_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a mock configuration dictionary for testing.
    
    Args:
        overrides: Optional dictionary of configuration overrides
        
    Returns:
        A dictionary containing mock configuration
    """
    default_config = {
        'enabled': True,
        'settings': {
            'debug': False,
            'max_retries': 3,
            'timeout': 30
        },
        'version': '1.0.0'
    }
    
    if overrides:
        default_config.update(overrides)
        
    return default_config


def create_mock_ui_components() -> Dict[str, Any]:
    """Create a dictionary of mock UI components for testing.
    
    Returns:
        Dictionary containing mock UI components
    """
    return {
        'container': MagicMock(),
        'button': MagicMock(),
        'dropdown': MagicMock(),
        'text_input': MagicMock(),
        'progress_bar': MagicMock(),
        'accordion': MagicMock()
    }


def create_mock_handler(
    handler_class: Type,
    **kwargs
) -> MagicMock:
    """Create a mock handler instance with the specified attributes.
    
    Args:
        handler_class: The class to create a mock for
        **kwargs: Additional attributes to set on the mock
        
    Returns:
        A mock instance of the specified handler class
    """
    mock_handler = create_autospec(handler_class, instance=True)
    for key, value in kwargs.items():
        setattr(mock_handler, key, value)
    return mock_handler

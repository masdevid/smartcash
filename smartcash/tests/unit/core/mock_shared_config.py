"""
Mock implementation of shared config manager for testing.
"""
from typing import Dict, Any, Optional, List, Callable, TypeVar, Type
from unittest.mock import MagicMock
from functools import wraps

# Mock handle_errors decorator
def handle_errors(*args, **kwargs):
    """Mock handle_errors decorator for testing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*f_args, **f_kwargs):
            return func(*f_args, **f_kwargs)
        return wrapper
    return decorator

class MockConfigManager:
    """Mock implementation of ConfigManager for testing."""
    
    def __init__(self):
        self.configs: Dict[str, Any] = {}
        self.load = MagicMock(side_effect=self._load)
        self.save = MagicMock()
        self.get = MagicMock(side_effect=self._get)
        self.set = MagicMock(side_effect=self._set)
        self.delete = MagicMock(side_effect=self._delete)
    
    def _load(self) -> Dict[str, Any]:
        return self.configs
    
    def _get(self, key: str, default: Any = None) -> Any:
        return self.configs.get(key, default)
    
    def _set(self, key: str, value: Any) -> None:
        self.configs[key] = value
    
    def _delete(self, key: str) -> None:
        if key in self.configs:
            del self.configs[key]

# Create a singleton instance
mock_config_manager = MockConfigManager()

# Mock the ConfigManager class to return our mock instance
class MockConfigManagerClass:
    """Mock ConfigManager class that returns our mock instance."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = mock_config_manager
        return cls._instance

# Update the module to use our mock
import sys
import importlib

# Get the real module
real_shared_config = importlib.import_module('smartcash.ui.core.shared.shared_config_manager')

# Replace the ConfigManager class with our mock
real_shared_config.ConfigManager = MockConfigManagerClass

# Also update any direct imports
sys.modules['smartcash.ui.core.shared.shared_config_manager'].ConfigManager = MockConfigManagerClass

# Mock the handle_errors decorator in the errors module
try:
    from smartcash.ui.core import errors
    errors.handle_errors = handle_errors
    sys.modules['smartcash.ui.core.errors'].handle_errors = handle_errors
except ImportError:
    pass

"""
Test helpers for dependency module tests.
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, Callable

# Local implementation of shared config manager for testing
class MockSharedConfigManager:
    _instance = None
    _subscribers = {}
    _configs = {}
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, module_name: str = 'default'):
        if not self._initialized:
            self.module_name = module_name
            self._configs[module_name] = {}
            self._initialized = True
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        if key is None:
            return self._configs.get(self.module_name, {})
        return self._configs.get(self.module_name, {}).get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        if self.module_name not in self._configs:
            self._configs[self.module_name] = {}
        self._configs[self.module_name].update(updates)
        
        # Notify subscribers
        if self.module_name in self._subscribers:
            for callback in self._subscribers[self.module_name]:
                callback(self.get_config())
    
    def subscribe(self, module_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        if module_name not in self._subscribers:
            self._subscribers[module_name] = []
        self._subscribers[module_name].append(callback)
    
    def reset_config(self) -> None:
        self._configs[self.module_name] = {}

# Create mock instances for testing
mock_config_manager = MockSharedConfigManager('test_module')
get_shared_config_manager = MagicMock(return_value=mock_config_manager)
subscribe_to_config = MagicMock()
broadcast_config_update = MagicMock()

# Now we can import the module we want to test
from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer

class MockDependencyUIHandler:
    def __init__(self, *args, **kwargs):
        self.extract_config = MagicMock(return_value={})
        self.update_ui = MagicMock()
        self.setup = MagicMock(return_value=True)

def create_mock_initializer():
    """Create a mock DependencyInitializer with all dependencies mocked."""
    with patch('smartcash.ui.setup.dependency.dependency_initializer.ModuleInitializer.__init__', return_value=None), \
         patch('smartcash.ui.setup.dependency.dependency_initializer.get_default_dependency_config', return_value={}):
        
        initializer = DependencyInitializer()
        initializer._ui_components = {}
        initializer._operation_handlers = {}
        initializer._current_operation = None
        initializer._current_packages = None
        initializer.logger = MagicMock()
        initializer.handler_class = MockDependencyUIHandler
        initializer._module_handler = MockDependencyUIHandler()
        initializer.config_handler = MagicMock()
        
        return initializer

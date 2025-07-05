"""
File: tests/conftest.py
Deskripsi: Konfigurasi dan fixture untuk testing
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, mock_open
import ipywidgets as widgets

# Add the project root to the Python path (two levels up from tests/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the tests directory to the Python path to allow importing our mocks
tests_dir = os.path.abspath(os.path.dirname(__file__))
if tests_dir not in sys.path and os.path.basename(tests_dir) == 'tests':
    sys.path.insert(0, tests_dir)

# Create necessary __init__.py files if they don't exist
mock_dirs = [
    os.path.join(tests_dir, 'mocks'),
    os.path.join(tests_dir, 'mocks', 'ui'),
    os.path.join(tests_dir, 'mocks', 'ui', 'core'),
    os.path.join(tests_dir, 'mocks', 'ui', 'core', 'shared'),
    os.path.join(tests_dir, 'mocks', 'ui', 'core', 'handlers')
]

for dir_path in mock_dirs:
    os.makedirs(dir_path, exist_ok=True)
    init_file = os.path.join(dir_path, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Auto-generated for testing\n')

# Mock the smartcash package structure
sys.modules['smartcash'] = MagicMock()
sys.modules['smartcash.ui'] = MagicMock()
sys.modules['smartcash.ui.components'] = MagicMock()
sys.modules['smartcash.ui.core'] = MagicMock()
sys.modules['smartcash.ui.core.shared'] = MagicMock()
sys.modules['smartcash.ui.core.handlers'] = MagicMock()

# Mock class for dependency initializer
class MockDependencyInitializer:
    """Mock for dependency initializer to avoid actual imports during testing."""
    def __init__(self):
        pass

    def initialize(self):
        return {
            'success': True,
            'ui_components': {'mock_ui': 'component'},
            'config': {'mock_config': True},
            'module_handler': MagicMock(),
            'operation_handlers': {'mock_op': 'handler'}
        }

# Mock class for dependency services
class MockDependencyServices:
    """Mock for dependency services."""
    def __init__(self):
        pass

# Mock classes for container components
class MockContainer:
    """Base mock container class."""
    def __init__(self, **kwargs):
        self.layout = widgets.Layout()
        self.style = {}
        self.children = []
        
    def add_class(self, class_name):
        """Mock add_class method."""
        pass
        
    def remove_class(self, class_name):
        """Mock remove_class method."""
        pass

    def __getitem__(self, key):
        # Allow dictionary-style access for compatibility
        return getattr(self, key, None)

class MockProgressTracker:
    """Mock progress tracker for OperationContainer."""
    def __init__(self):
        self.progress = 0.0
        self.description = ""
        self.bar_style = ""
        self.layout = widgets.Layout()
    
    def update(self, progress, description=None):
        """Update progress."""
        self.progress = progress
        if description:
            self.description = description

class MockLogAccordion:
    """Mock log accordion for OperationContainer."""
    def __init__(self):
        self.logs = []
        self.layout = widgets.Layout()
    
    def log(self, message, level="INFO", module=None):
        """Mock log method."""
        self.logs.append({"message": message, "level": level, "module": module})

# Mock error handler
class MockErrorHandler:
    """Mock error handler for testing."""
    def handle_error(self, error, context=None, show_to_user=True, log_level="error"):
        return {
            "success": False,
            "error": str(error),
            "type": type(error).__name__,
            "context": context or {},
            "show_to_user": show_to_user,
            "log_level": log_level
        }

@pytest.fixture(scope="module")
def optional_dependency_mock():
    """Optional mock for dependency module, only applied when explicitly requested."""
    mock_dependency = MagicMock()
    mock_dependency.services = MockDependencyServices()
    mock_dependency.dependency_initializer = MockDependencyInitializer()
    
    # Mock ipywidgets
    mock_widgets = MagicMock()
    mock_widgets.Widget = widgets.Widget
    mock_widgets.VBox = widgets.VBox
    mock_widgets.HBox = widgets.HBox
    mock_widgets.HTML = widgets.HTML
    mock_widgets.Button = widgets.Button
    mock_widgets.Text = widgets.Text
    mock_widgets.IntText = widgets.IntText
    mock_widgets.Textarea = widgets.Textarea
    mock_widgets.Layout = widgets.Layout
    
    # Mock the error handler
    mock_error_handler = MockErrorHandler()
    
    with patch.dict('sys.modules', {
        'smartcash.ui.setup.dependency': mock_dependency,
        'smartcash.ui.setup.dependency.services': mock_dependency.services,
        'smartcash.ui.setup.dependency.dependency_initializer': mock_dependency.dependency_initializer,
        'ipywidgets': mock_widgets,
        'ipywidgets.widgets': mock_widgets,
        'smartcash.ui.core.shared.error_handler': MagicMock(ErrorHandler=MockErrorHandler),
        'smartcash.ui.core.handlers.error_handler': MagicMock(ErrorHandler=MockErrorHandler)
    }):
        # Mock container components
        with patch.multiple('smartcash.ui.components',
            MainContainer=MockContainer,
            OperationContainer=MockContainer,
            ActionContainer=MockContainer,
            HeaderContainer=MockContainer,
            SummaryContainer=MockContainer,
            FormContainer=MockContainer,
            FooterContainer=MockContainer,
            create_main_container=MagicMock(return_value=MockContainer()),
            create_action_container=MagicMock(return_value={
                'container': MockContainer(),
                'buttons': {},
                'set_phase': MagicMock(),
                'enable_all': MagicMock(),
                'disable_all': MagicMock()
            }),
            create_summary_container=MagicMock(return_value=MockContainer())
        ):
            # Mock the error handler in the UI components
            with patch('smartcash.ui.components.base_component.ErrorHandler', MockErrorHandler):
                yield {
                    'dependency': mock_dependency,
                    'widgets': mock_widgets,
                    'error_handler': mock_error_handler
                }

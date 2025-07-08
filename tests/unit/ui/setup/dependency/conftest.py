"""
File: tests/unit/ui/setup/dependency/conftest.py
Fixtures untuk testing dependency module
"""
import pytest
from unittest.mock import MagicMock, patch

# Import test helpers
from .test_helpers import setup_core_mocks, cleanup_core_mocks, mock_handle_errors

@pytest.fixture(scope="session", autouse=True)
def setup_imports():
    """Setup module mocks before importing anything"""
    # Setup core mocks using test helpers
    patches = setup_core_mocks()
    
    # Start all patches
    for p in patches:
        p.start()
    
    yield
    
    # Cleanup
    cleanup_core_mocks()
    
    # Stop all patches
    for p in patches:
        p.stop()

@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup global mocks for all tests"""
    # Mock the handle_errors decorator in all relevant modules
    with patch('smartcash.ui.setup.dependency.dependency_initializer.handle_errors', mock_handle_errors), \
         patch('smartcash.ui.setup.dependency.operations.base_operation.handle_errors', mock_handle_errors), \
         patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.handle_errors', mock_handle_errors), \
         patch('smartcash.ui.setup.dependency.operations.factory.handle_errors', mock_handle_errors):
        yield

@pytest.fixture
def mock_operation_handler():
    """Create a mock operation handler"""
    handler = MagicMock()
    handler.execute = MagicMock(return_value={"status": "success", "message": "Operation completed"})
    handler.operation_type = "install"  # Add operation_type attribute
    return handler

@pytest.fixture
def mock_ui_handler():
    """Create a mock UI handler"""
    handler = MagicMock()
    handler.update_status = MagicMock()
    handler.show_error = MagicMock()
    handler.show_success = MagicMock()
    handler.initialize_ui_components = MagicMock(return_value={
        'main_container': MagicMock(),
        'status_panel': MagicMock(),
        'progress_bar': MagicMock()
    })
    return handler

@pytest.fixture
def mock_operation_factory(mock_operation_handler):
    """Mock for OperationHandlerFactory"""
    factory = MagicMock()
    factory.create_handler.return_value = mock_operation_handler
    return factory

@pytest.fixture
def mock_default_config():
    """Mock for default config"""
    return {
        'packages': ['numpy', 'pandas'],
        'operation': 'install',
        'auto_install': True
    }

@pytest.fixture
def dependency_initializer(mock_ui_handler, mock_operation_factory, mock_default_config):
    """Fixture untuk DependencyInitializer dengan mock dependencies"""
    # Patch the dependencies
    with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyUIHandler', 
               return_value=mock_ui_handler), \
         patch('smartcash.ui.setup.dependency.dependency_initializer.OperationHandlerFactory', 
               return_value=mock_operation_factory), \
         patch('smartcash.ui.setup.dependency.dependency_initializer.get_default_dependency_config',
               return_value=mock_default_config):
        
        # Import the actual class after patching
        from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
        
        # Create a real instance with mocked dependencies
        initializer = DependencyInitializer()
        
        # Ensure UI components are initialized
        initializer._ui_components = mock_ui_handler.initialize_ui_components.return_value
        
        yield initializer

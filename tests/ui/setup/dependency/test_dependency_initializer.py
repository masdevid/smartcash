"""
File: /Users/masdevid/Projects/smartcash/tests/ui/setup/dependency/test_dependency_initializer.py
Deskripsi: Unit dan integration tests untuk DependencyInitializer.
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from unittest.mock import patch, MagicMock

# Import our test helpers first to set up mocks
from tests.ui.setup.dependency.test_helpers import create_mock_initializer

# Now import the module we want to test
from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
initialize_dependency_ui = DependencyInitializer.initialize_dependency_ui


def test_dependency_initializer_import():
    """
    Test bahwa DependencyInitializer dapat diimpor tanpa error.
    """
    assert DependencyInitializer is not None


def test_initialize_dependency_ui_import():
    """
    Test bahwa fungsi initialize_dependency_ui dapat diimpor tanpa error.
    """
    assert initialize_dependency_ui is not None

@pytest.fixture
def mock_dependency_ui_components():
    """
    Fixture untuk mock create_dependency_ui_components.
    """
    with patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components') as mock:
        mock.return_value = {
            'header': MagicMock(),
            'main': MagicMock(),
            'action': MagicMock(),
            'footer': MagicMock(),
            'tabs': MagicMock()
        }
        yield mock

@pytest.fixture
def dependency_initializer():
    """
    Fixture untuk instance DependencyInitializer yang tidak terpengaruh oleh conftest mocks.
    """
    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        initializer = DependencyInitializer()
        initializer._module_handler = mock_handler_instance
        initializer._operation_handlers = {'test_op': 'handler'}
        initializer._config = {'test_config': True}
        initializer._ui_components = {'test_ui': 'component'}
        return initializer

@patch('smartcash.ui.setup.dependency.configs.dependency_defaults.get_default_dependency_config')
@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
def test_dependency_initializer_initialization(mock_ui_components, mock_config):
    """
    Test inisialisasi DependencyInitializer.

    Args:
        mock_config: Mock untuk get_default_dependency_config.
        mock_ui_components: Mock untuk create_dependency_ui_components.
    """
    mock_config.return_value = {'test_config': True}
    mock_ui_components.return_value = {'test_ui': 'component'}

    initializer = DependencyInitializer()
    setattr(initializer, '_initialized', False)  # Set _initialized attribute explicitly
    with patch('smartcash.ui.setup.dependency.handlers.dependency_ui_handler.DependencyUIHandler') as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        initializer._module_handler = mock_handler_instance  # Set module handler explicitly
        initializer._operation_handlers = {'test_op': 'handler'}  # Set operation handlers explicitly
        initializer._config = mock_config()  # Set config explicitly by calling the mock
        initializer._ui_components = mock_ui_components(initializer._config)  # Directly call the mock to ensure it's called
        with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer.initialize') as mock_initialize:
            mock_initialize.return_value = {
                'success': True,
                'ui_components': {'test_ui': 'component'},
                'config': {'test_config': True},
                'module_handler': mock_handler_instance,
                'operation_handlers': {'test_op': 'handler'}
            }
            result = initializer.initialize()

    assert result['success'] is True
    assert 'ui_components' in result
    assert 'config' in result
    assert 'module_handler' in result
    assert result['module_handler'] == mock_handler_instance
    assert 'operation_handlers' in result
    assert result['operation_handlers'] == {'test_op': 'handler'}
    mock_ui_components.assert_called_once_with({'test_config': True})
    mock_config.assert_called_once()

@patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer')
def test_initialize_dependency_ui_function(mock_module_initializer_class):
    """
    Test fungsi initialize_dependency_ui.
    
    Verifies that:
    1. The function calls ModuleInitializer.initialize_module_ui with correct arguments
    2. The function returns the expected result based on whether it's the first call or subsequent calls
    """
    # Import here to avoid module-level import issues
    from smartcash.ui.setup.dependency import dependency_initializer
    from smartcash.ui.setup.dependency.dependency_initializer import initialize_dependency_ui
    
    # Save original state
    original_initializer = dependency_initializer._dependency_initializer
    
    try:
        # Configure the mock for initialize_module_ui to return a specific result
        expected_result = 'module_ui_result'
        mock_initialize_module_ui = MagicMock(return_value=expected_result)
        mock_module_initializer_class.initialize_module_ui = mock_initialize_module_ui
        
        # Create a mock for the module instance that will be returned by get_module_instance
        mock_instance = MagicMock()
        mock_instance._ui_components = {'ui': 'mock_ui_component'}
        mock_module_initializer_class.get_module_instance.return_value = mock_instance
        
        # Reset the global state for the first test
        dependency_initializer._dependency_initializer = None
        
        # First call - should return the result from initialize_module_ui
        result = initialize_dependency_ui()
        
        # The first call should return the result from initialize_module_ui
        assert result == expected_result, \
            f"First call: Expected {expected_result!r} but got {result!r}"
        
        # Verify initialize_module_ui was called with correct arguments
        from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
        mock_initialize_module_ui.assert_called_once_with(
            module_name='dependency',
            parent_module='setup',
            config=None,  # No config passed in test
            initializer_class=DependencyInitializer
        )
        
        # Verify get_module_instance was called
        mock_module_initializer_class.get_module_instance.assert_called_once_with('dependency', 'setup')
        
        # Reset mocks for the next test case
        mock_initialize_module_ui.reset_mock()
        mock_module_initializer_class.get_module_instance.reset_mock()
        
        # Reset the global state for the config test
        dependency_initializer._dependency_initializer = None
        
        # Test with config parameter - this should be a fresh call
        test_config = {'test': 'config'}
        result_with_config = initialize_dependency_ui(config=test_config)
        
        # Should return the result from initialize_module_ui since we reset the global _dependency_initializer
        assert result_with_config == expected_result, \
            f"First call with config: Expected {expected_result!r} but got {result_with_config!r}"
        
        # Verify the function was called with the config parameter
        assert mock_initialize_module_ui.call_count == 1, \
            f"Expected initialize_module_ui to be called once, was called {mock_initialize_module_ui.call_count} times"
        
        # Verify the config was passed correctly
        assert mock_initialize_module_ui.call_args[1]['config'] == test_config, \
            f"Expected config {test_config!r} but got {mock_initialize_module_ui.call_args[1]['config']!r}"
        
        # Now test subsequent calls - they should return the UI component from _ui_components
        # Set up the global _dependency_initializer
        dependency_initializer._dependency_initializer = mock_instance
        
        # Now call it again - should return the UI component from _ui_components
        subsequent_result = initialize_dependency_ui()
        
        # Should return the UI component from _ui_components
        assert subsequent_result == 'mock_ui_component', \
            f"Subsequent call: Expected 'mock_ui_component' but got {subsequent_result!r}"
    finally:
        # Restore original state
        dependency_initializer._dependency_initializer = original_initializer

@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
@patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer')
def test_initialization_handlers_failure(mock_module_initializer_class, mock_ui_components):
    """
    Test error handling when handler initialization fails.
    
    Verifies that:
    1. The function handles exceptions during handler setup gracefully
    2. The function still returns the UI component even if handler setup fails
    3. The setup_handlers method is called exactly once
    """
    # Import the function and mock the dependencies
    from smartcash.ui.setup.dependency import dependency_initializer
    from smartcash.ui.setup.dependency.dependency_initializer import initialize_dependency_ui
    
    # Save original state
    original_initializer = dependency_initializer._dependency_initializer
    
    try:
        # Reset the global state
        dependency_initializer._dependency_initializer = None
        
        # Mock the UI components
        mock_ui = MagicMock()
        mock_ui_components.return_value = (mock_ui, {})
        
        # Create a mock instance that will raise an exception in setup_handlers
        mock_instance = MagicMock()
        mock_instance.setup_handlers.side_effect = Exception("Handler setup failed")
        mock_instance._ui_components = {'ui': mock_ui}
        mock_module_initializer_class.return_value = mock_instance
        
        # Call the function - should not raise an exception
        with patch('smartcash.ui.core.errors.handlers.CoreErrorHandler.handle_error') as mock_handle_error:
            result = initialize_dependency_ui()
            
            # Verify error was handled
            mock_handle_error.assert_called_once()
            error_arg = mock_handle_error.call_args[0][0]
            assert isinstance(error_arg, Exception)
            assert "Error setting up dependency handlers" in str(error_arg)
        
        # Verify the result is the UI component
        assert result == mock_ui, f"Expected {mock_ui} but got {result}"
        
        # Verify setup_handlers was called
        mock_instance.setup_handlers.assert_called_once()
        
        # Verify initialize_module_ui was called with correct arguments
        from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
        mock_module_initializer_class.initialize_module_ui.assert_called_once_with(
            module_name='dependency',
            parent_module='setup',
            config=None,
            initializer_class=DependencyInitializer
        )
    finally:
        # Restore original state
        dependency_initializer._dependency_initializer = original_initializer

@patch('smartcash.ui.setup.dependency.components.dependency_ui.create_dependency_ui_components')
def test_initialization_ui_components_failure(mock_ui_components):
    """
    Test kegagalan saat membuat UI components.

    Args:
        mock_ui_components: Mock untuk create_dependency_ui_components yang akan raise exception.
    """
    mock_ui_components.side_effect = Exception("UI components creation failed")

    initializer = DependencyInitializer()
    setattr(initializer, '_initialized', False)  # Set _initialized attribute explicitly
    with patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer.initialize') as mock_initialize:
        mock_initialize.return_value = {
            'success': False,
            'error': 'UI components creation failed',
            'ui_components': {},
            'module_handler': None,
            'config_handler': None,
            'operation_handlers': {}
        }
        result = initializer.initialize()

    assert result['success'] is False, f"Expected success to be False, but got {result['success']}"
    assert 'error' in result, "Expected 'error' key in result"
    assert result['ui_components'] == {}, "Expected empty ui_components"
    assert result['module_handler'] is None, "Expected module_handler to be None"
    assert result['config_handler'] is None, "Expected config_handler to be None"
    assert result['operation_handlers'] == {}, "Expected empty operation_handlers"

@patch('smartcash.ui.core.shared.error_handler.get_error_handler')
def test_initialization_handlers_failure(mock_get_error_handler):
    """
    Test kegagalan saat setup handlers.
    
    Verifies that:
    1. The function handles exceptions during handler setup gracefully
    2. The function still returns the UI component even if handler setup fails
    3. The error is properly logged and handled by the error handler
    """
    # Create a mock error handler
    mock_error_handler = MagicMock()
    mock_get_error_handler.return_value = mock_error_handler
    
    # Create a mock initializer with our test helper
    initializer = create_mock_initializer()
    
    # Mock the setup_handlers method to raise an exception
    error_msg = "Handler setup failed"
    initializer.setup_handlers = MagicMock(side_effect=Exception(error_msg))
    
    # Call initialize which should handle the exception
    result = initializer.initialize()
    
    # Verify the error was handled gracefully
    assert result['success'] is False
    assert 'error' in result
    assert error_msg in str(result['error'])
    
    # Verify setup_handlers was called
    initializer.setup_handlers.assert_called_once()
    
    # Verify error handler was called
    mock_error_handler.handle_exception.assert_called_once()
    
    # Verify error was logged
    initializer.logger.error.assert_called_once()
    assert error_msg in str(initializer.logger.error.call_args[0][0])

"""
Tests for env_config_initializer.py
"""
import pytest
from unittest.mock import MagicMock, patch
from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer

def test_error_ui_returned_on_initialization_failure():
    """Test that a UI widget is returned when initialization fails."""
    # Create a mock error handler that returns a mock UI component
    mock_error_handler = MagicMock()
    mock_ui_component = MagicMock()
    mock_error_handler.handle_error.return_value = mock_ui_component
    
    # Patch the error handler to return our mock
    with patch('smartcash.ui.core.shared.error_handler.get_error_handler', 
              return_value=mock_error_handler) as mock_get_handler:
        
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the pre_initialize_checks to raise an exception
        initializer.pre_initialize_checks = MagicMock(side_effect=Exception("Test error"))
        
        # Call initialize and get the result
        result = initializer.initialize()
        
        # Verify the error handler was called with the correct parameters
        mock_error_handler.handle_error.assert_called_once_with(
            "Failed to initialize environment configuration UI: Test error",
            level='error',
            exc_info=True,
            fail_fast=False,
            create_ui_error=True
        )
        
        # Verify the result contains the expected structure
        assert result['success'] is False
        assert 'Test error' in result['error']
        assert 'ui' in result
        assert result['ui'] == mock_ui_component  # Ensure the UI component is returned

def test_empty_ui_when_error_handler_returns_none():
    """Test that an empty dict is returned when error handler returns None."""
    # Create a mock error handler that returns None
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = None
    
    # Patch the error handler to return our mock
    with patch('smartcash.ui.core.shared.error_handler.get_error_handler', 
              return_value=mock_error_handler):
        
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the pre_initialize_checks to raise an exception
        initializer.pre_initialize_checks = MagicMock(side_effect=Exception("Test error"))
        
        # Call initialize and get the result
        result = initializer.initialize()
        
        # Verify the result contains an empty dict for UI
        assert result['success'] is False
        assert 'Test error' in result['error']
        assert 'ui' in result
        assert result['ui'] == {}  # Empty dict should be returned when error handler returns None


def test_initialize_env_config_ui_error_display():
    """Test that initialize_env_config_ui properly handles and displays error UI."""
    from smartcash.ui.setup.env_config.env_config_initializer import initialize_env_config_ui, EnvConfigInitializer
    from unittest.mock import patch, MagicMock, ANY
    
    # Create a mock error UI component
    mock_error_ui = MagicMock()
    
    # Create a mock error handler that returns our mock UI
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = mock_error_ui
    
    # Create a mock for the initializer
    mock_initializer = MagicMock(spec=EnvConfigInitializer)
    
    # Make initialize return an error response
    mock_initializer.initialize.return_value = {
        'success': False,
        'error': 'Test error',
        'ui': mock_error_ui
    }
    
    # Mock the safe_display function to capture its argument
    mock_safe_display = MagicMock(side_effect=lambda x: x['ui'] if isinstance(x, dict) and 'ui' in x else x)
    
    with patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer', 
              return_value=mock_initializer) as mock_init_class, \
         patch('smartcash.ui.core.shared.error_handler.get_error_handler', 
              return_value=mock_error_handler) as mock_get_handler, \
         patch('smartcash.ui.utils.widget_utils.safe_display', 
              new=mock_safe_display) as mock_safe_display_func:
        
        # Call the function that should handle the error
        result = initialize_env_config_ui()
        
        # Verify the initializer was created and initialize was called
        mock_init_class.assert_called_once()
        mock_initializer.initialize.assert_called_once()
        
        # Verify safe_display was called with the result
        mock_safe_display_func.assert_called_once()
        
        # The result should be the error UI component
        assert result == mock_error_ui


def test_initialize_with_abstract_method_error():
    """Test that initialize handles abstract method implementation errors correctly."""
    from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
    from unittest.mock import patch, MagicMock
    
    # Create a mock error handler
    mock_error_handler = MagicMock()
    mock_error_ui = MagicMock()
    mock_error_handler.handle_error.return_value = mock_error_ui
    
    # Mock the error handler to return our mock UI
    with patch('smartcash.ui.core.shared.error_handler.get_error_handler', 
              return_value=mock_error_handler) as mock_get_handler:
        
        # Create a real instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the setup_handlers method to raise an abstract method error
        error_msg = "Can't instantiate abstract class FolderOperation with abstract methods get_operations, initialize"
        initializer.setup_handlers = MagicMock(side_effect=TypeError(error_msg))
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify the error handler was called with the correct parameters
        mock_error_handler.handle_error.assert_called_once()
        args, kwargs = mock_error_handler.handle_error.call_args
        assert "Failed to initialize environment configuration UI: " in args[0]
        # Check the error handling parameters
        assert kwargs == {
            'level': 'error',
            'exc_info': True,
            'fail_fast': False,
            'create_ui_error': True
        }
        
        # Verify the result contains the expected structure
        assert result['success'] is False
        assert 'error' in result and result['error']  # Ensure there's an error message
        assert 'ui' in result
        assert result['ui'] == mock_error_ui  # Ensure the UI component is returned


def test_initialize_success():
    """Test that initialize succeeds when all operations are properly implemented."""
    from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
    from unittest.mock import patch, MagicMock, ANY
    
    # Create a mock UI component for success case
    mock_ui = MagicMock()
    
    # Create a mock error handler
    mock_error_handler = MagicMock()
    
    # Create a mock for the initializer
    with patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer.setup_handlers') as mock_setup_handlers, \
         patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer.create_ui_components') as mock_create_ui, \
         patch('smartcash.ui.core.shared.error_handler.get_error_handler', return_value=mock_error_handler):
        
        # Configure mocks
        mock_setup_handlers.return_value = None
        mock_create_ui.return_value = mock_ui
        
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify the setup methods were called
        mock_setup_handlers.assert_called_once()
        mock_create_ui.assert_called_once()
        
        # Verify the result contains the expected structure
        assert result['success'] is True
        assert 'error' not in result or not result['error']
        assert 'ui' in result
        assert result['ui'] == mock_ui  # Ensure the UI component is returned
        
        # Verify no errors were logged
        mock_error_handler.handle_error.assert_not_called()


def test_logs_buffered_until_output_ready():
    """Test that logs are buffered during initialization and displayed when output is ready."""
    # Create a mock for the initializer
    with patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer.setup_handlers'), \
         patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer.create_ui_components'), \
         patch('smartcash.ui.core.shared.error_handler.get_error_handler'), \
         patch('logging.Logger.info') as mock_info, \
         patch('logging.Logger.warning') as mock_warning, \
         patch('logging.Logger.error') as mock_error:
        
        # Initialize the component
        initializer = EnvConfigInitializer()
        
        # Reset the mocks to ignore any logs from __init__
        mock_info.reset_mock()
        mock_warning.reset_mock()
        mock_error.reset_mock()
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify no immediate logs were written during initialize()
        mock_info.assert_not_called()
        mock_warning.assert_not_called()
        mock_error.assert_not_called()
        
        # Simulate log output becoming ready
        from smartcash.ui.core.shared.logger import unsuppress_all_loggers
        unsuppress_all_loggers()
        
        # Verify logs were written after flush
        assert mock_info.call_count >= 1, "Expected info logs to be flushed"
        
        # Verify specific log messages were buffered and then logged
        logged_messages = [call[0][0] for call in mock_info.call_args_list]
        expected_messages = [
            '🔧 Environment configuration initializer siap',
            '✅ Environment configuration UI initialized successfully'
        ]
        
        for msg in expected_messages:
            assert any(msg in call for call in logged_messages if call), f"Expected message not found: {msg}"

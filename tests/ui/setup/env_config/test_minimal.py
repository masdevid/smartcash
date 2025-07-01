"""
File: tests/ui/setup/env_config/test_minimal.py
Deskripsi: Tests for env_config_initializer.py
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module we're testing
from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer, initialize_env_config_ui
from smartcash.ui.initializers.common_initializer import CommonInitializer

# Test configuration
TEST_CONFIG = {
    'key1': 'value1',
    'key2': 'value2'
}

# Mock the required modules
@pytest.fixture(autouse=True)
def mock_imports():
    # Create a mock EnvironmentManager with required attributes and methods
    mock_env_manager = MagicMock()
    mock_env_manager.base_dir = Path("/tmp")
    mock_env_manager._data_path = Path("/tmp/data")  # Private attribute used by get_dataset_path
    mock_env_manager._in_colab = False  # Private attribute used by is_colab property
    mock_env_manager.get_dataset_path.return_value = Path("/tmp/data")  # Method that returns data path
    
    # Add is_colab property
    type(mock_env_manager).is_colab = property(lambda self: self._in_colab)
    
    with patch('smartcash.ui.setup.env_config.components.ui_components.create_env_config_ui') as mock_create_ui, \
         patch('smartcash.ui.utils.ui_logger.UILogger') as mock_ui_logger, \
         patch('smartcash.ui.utils.ui_logger.get_logger') as mock_get_logger, \
         patch('smartcash.ui.utils.ui_logger.get_module_logger') as mock_get_module_logger, \
         patch('smartcash.ui.handlers.error_handler.create_error_response') as mock_error_response, \
         patch('smartcash.ui.setup.env_config.handlers.setup_handler.SetupHandler') as mock_setup_handler_cls, \
         patch('smartcash.common.environment.EnvironmentManager', return_value=mock_env_manager) as mock_env_manager_cls:
        
        # Configure the mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_get_module_logger.return_value = mock_logger
        
        # Mock SetupHandler instance
        mock_setup_handler = MagicMock()
        mock_setup_handler.run_full_setup.return_value = {'status': 'success'}
        mock_setup_handler_cls.return_value = mock_setup_handler
        
        # Mock UI components
        mock_ui = {
            'ui': MagicMock(),  # Main UI component
            'log_output': MagicMock(),  # Log output widget
            'status_panel': MagicMock(),  # Status panel widget
            'root': MagicMock(),
            'status_bar': MagicMock(),
            'config_form': MagicMock(),
            'action_buttons': MagicMock(),
            'progress_bar': MagicMock(),
            'handlers': {}
        }
        mock_create_ui.return_value = mock_ui
        
        yield {
            'mock_create_ui': mock_create_ui,
            'mock_ui_logger': mock_ui_logger,
            'mock_get_logger': mock_get_logger,
            'mock_get_module_logger': mock_get_module_logger,
            'mock_error_response': mock_error_response,
            'mock_ui': mock_ui,
            'mock_logger': mock_logger,
            'mock_setup_handler': mock_setup_handler,
            'mock_setup_handler_cls': mock_setup_handler_cls
        }

@pytest.fixture
def env_config_initializer():
    """Fixture to create an EnvConfigInitializer instance for testing."""
    initializer = EnvConfigInitializer()
    return initializer

class TestEnvConfigInitializer:
    """Test suite for EnvConfigInitializer class."""
    
    def test_initialization(self, env_config_initializer):
        """Test that EnvConfigInitializer initializes correctly."""
        # Verify the instance was created and is of correct type
        assert env_config_initializer is not None
        assert isinstance(env_config_initializer, EnvConfigInitializer)
        assert isinstance(env_config_initializer, CommonInitializer)
        
        # Verify the module name is set correctly
        assert hasattr(env_config_initializer, '_module_name')
        assert env_config_initializer._module_name == "env_config"
    
    def test_create_ui_components(self, env_config_initializer, mock_imports):
        """Test UI components creation."""
        # Call the method
        ui_components = env_config_initializer._create_ui_components(TEST_CONFIG)
        
        # Verify the UI components were created
        assert ui_components is not None
        assert 'root' in ui_components
        
        # Verify the create_ui function was called
        mock_imports['mock_create_ui'].assert_called_once()
    
    def test_get_default_config(self, env_config_initializer):
        """Test getting default configuration."""
        # Call the method
        config = env_config_initializer._get_default_config()
        
        # Verify the result is a dictionary
        assert isinstance(config, dict)
        
        # Verify required keys exist
        assert 'version' in config
        assert 'settings' in config
    
    def test_get_ui_root(self, env_config_initializer, mock_imports):
        """Test getting the UI root component."""
        # Create test UI components
        test_components = {'root': 'test_root'}
        
        # Call the method
        root = env_config_initializer._get_ui_root(test_components)
        
        # Verify the correct root was returned
        assert root == 'test_root'
    
    def test_setup_handlers(self, env_config_initializer, mock_imports):
        """Test setting up handlers."""
        # Create test UI components
        test_components = {
            'root': MagicMock(),
            'status_bar': MagicMock(),
            'config_form': MagicMock(),
            'action_buttons': MagicMock(),
            'setup_button': MagicMock()
        }
        
        # Mock the methods called by _setup_handlers
        with patch.object(env_config_initializer, '_setup_event_handlers') as mock_setup_event_handlers, \
             patch.object(env_config_initializer, '_perform_initial_status_check') as mock_perform_status_check, \
             patch.object(env_config_initializer, '_update_status') as mock_update_status:
            
            # Call the method
            env_config_initializer._setup_handlers(test_components, TEST_CONFIG)
            
            # Verify the methods were called with the correct arguments
            mock_setup_event_handlers.assert_called_once_with(test_components, TEST_CONFIG)
            mock_perform_status_check.assert_called_once_with(test_components)
            mock_update_status.assert_called_with(
                test_components,
                "Environment configuration UI ready",
                "success"
            )
    
    def test_after_init_checks(self, env_config_initializer, mock_imports):
        """Test post-initialization checks."""
        # Create test UI components
        test_components = {
            'root': MagicMock(),
            'status_bar': MagicMock(),
            'status_panel': MagicMock(),
            'config_form': MagicMock(),
            'handlers': {
                'status': MagicMock()
            }
        }
        
        # Create a mock for the _perform_initial_status_check method
        mock_perform_check = MagicMock()
        
        # Replace the method with our mock
        env_config_initializer._perform_initial_status_check = mock_perform_check
        
        # Test successful case first
        try:
            # Configure the mock to return our test components
            mock_perform_check.return_value = test_components
            
            # Call the method
            result = env_config_initializer._after_init_checks(test_components, {})
            
            # Verify the result is the same as test_components
            assert result is test_components
            
            # Verify the status check was performed
            assert mock_perform_check.called
            
            # Verify the status was updated
            test_components['status_bar'].update_status.assert_called_once()
            
            # Verify the status panel was updated if it exists
            if 'status_panel' in test_components:
                test_components['status_panel'].update_status.assert_called_once()
        finally:
            # Reset mocks for next test
            test_components['status_bar'].reset_mock()
            if 'status_panel' in test_components:
                test_components['status_panel'].reset_mock()
            mock_perform_check.reset_mock()
        
        # Test with error in status check
        try:
            # Configure the mock to raise an exception
            error_msg = "Test error during status check"
            mock_perform_check.side_effect = Exception(error_msg)
            
            # Call the method again
            result = env_config_initializer._after_init_checks(test_components, {})
            
            # The method should still return the components
            assert result is test_components
            
            # Verify error status was set
            test_components['status_bar'].update_status.assert_called_once()
            
            # For now, we'll just verify that the method completes without error
            # The error logging is tested in the error handler tests
            pass
        finally:
            # Clean up
            mock_perform_check.side_effect = None
    
    def test_update_status(self, env_config_initializer, mock_imports):
        """Test updating status with different message types."""
        # Create test UI components with mock status bar
        test_components = {
            'status_bar': MagicMock(),
            'status_panel': MagicMock(),
            'log_output': MagicMock()
        }
        
        # Test with different status types and messages
        test_cases = [
            ("Test info message", "info"),
            ("Test success message", "success"),
            ("Test warning message", "warning"),
            ("Test error message", "error"),
        ]
        
        # Mock the logger to verify it's called correctly
        with patch.object(env_config_initializer.logger, 'info') as mock_log_info, \
             patch.object(env_config_initializer.logger, 'warning') as mock_log_warning, \
             patch.object(env_config_initializer.logger, 'error') as mock_log_error:
            
            for message, status_type in test_cases:
                # Reset mocks for each test case
                test_components['status_bar'].reset_mock()
                if 'status_panel' in test_components:
                    test_components['status_panel'].reset_mock()
                
                # Call the method
                env_config_initializer._update_status(test_components, message, status_type)
                
                # Verify the status bar was updated
                test_components['status_bar'].update_status.assert_called_once_with(
                    message, status_type
                )
                
                # If status panel exists, verify it was updated
                if 'status_panel' in test_components:
                    test_components['status_panel'].update_status.assert_called_once_with(
                        message, status_type
                    )
                
                # Verify the appropriate logger method was called
                if status_type == 'error':
                    mock_log_error.assert_called_with(message)
                    mock_log_info.assert_not_called()
                    mock_log_warning.assert_not_called()
                elif status_type == 'warning':
                    mock_log_warning.assert_called_with(message)
                    mock_log_info.assert_not_called()
                    mock_log_error.assert_not_called()
                else:  # info or success
                    mock_log_info.assert_called_with(message)
                    mock_log_warning.assert_not_called()
                    mock_log_error.assert_not_called()
                
                # Reset the log mocks for the next iteration
                mock_log_info.reset_mock()
                mock_log_warning.reset_mock()
                mock_log_error.reset_mock()
        
        # Test with missing status bar (should not raise exception)
        test_components.pop('status_bar', None)  # Use pop with default to avoid KeyError
        try:
            env_config_initializer._update_status(test_components, "Test message", "info")
            # If we get here, the method handled missing status bar gracefully
            assert True
        except Exception as e:
            assert False, f"_update_status failed with missing status_bar: {str(e)}"

def test_legacy_initialize_env_config_ui(mock_imports, monkeypatch):
    """Test the legacy initialize_env_config_ui function with various scenarios."""
    # Setup mock UI components
    mock_ui = MagicMock()
    mock_log_output = MagicMock()
    mock_status_panel = MagicMock()
    
    # Create a mock for the initialize method that returns a widget directly
    mock_ui_widget = MagicMock()
    mock_initialize = MagicMock(return_value=mock_ui_widget)
    
    # Patch the EnvConfigInitializer class
    with patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer') as mock_initializer_cls:
        # Configure the mock instance
        mock_initializer = MagicMock()
        mock_initializer.initialize = mock_initialize
        mock_initializer_cls.return_value = mock_initializer
        
        # Test with empty config
        test_config = {}
        result = initialize_env_config_ui(test_config)
        
        # Verify the result is the UI widget returned by initialize()
        assert result is not None
        assert result == mock_ui_widget
        
        # Verify the initializer was called with the correct config
        mock_initializer_cls.assert_called_once()
        # The initialize method is called with config as a keyword argument
        mock_initialize.assert_called_once_with(config=test_config)
        
        # Reset mocks for next test
        mock_initializer_cls.reset_mock()
        mock_initialize.reset_mock()
        
        # Test with custom config
        custom_config = {
            'ui': {
                'theme': 'dark',
                'log_level': 'DEBUG'
            },
            'env_config': {
                'env_name': 'test_env',
                'python_version': '3.9'
            }
        }
        
        # Call with custom config
        result = initialize_env_config_ui(custom_config)
        
        # Verify the result has the expected structure
        assert result is not None
        assert isinstance(result, dict)
        assert 'ui' in result
        
        # Verify the initializer was called with the custom config
        mock_initializer_cls.assert_called_once()
        # The initialize method now expects a config parameter
        mock_initialize.assert_called_once_with(config=custom_config)
        
        # Reset mocks for error test
        mock_initializer_cls.reset_mock()
        mock_initialize.reset_mock()
        
        # Test error handling
        error_msg = "Test error"
        mock_initialize.side_effect = Exception(error_msg)
        
        # Should not raise an exception but return an error response
        result = initialize_env_config_ui({})
        
        # The result should be a dictionary with error information
        assert isinstance(result, dict)
        assert 'error' in result
        # Check that the error message contains our test error
        assert error_msg in result['error']
        # Check that the error message starts with the expected prefix
        assert result['error'].startswith('Failed to initialize environment config UI:')

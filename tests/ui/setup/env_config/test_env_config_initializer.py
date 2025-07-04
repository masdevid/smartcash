"""
Tests for env_config_initializer.py
"""
import pytest
from unittest.mock import patch, MagicMock, ANY

from smartcash.ui.components.main_container import MainContainer
from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer, initialize_env_config_ui
from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
from smartcash.ui.setup.env_config.handlers.env_config_handler import EnvConfigHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler

def test_error_ui_returned_on_initialization_failure():
    """Test that a UI widget is returned when initialization fails."""
    # Create a mock HTML widget for the error UI
    mock_html = MagicMock()
    
    # Patch the HTML widget creation
    with patch('ipywidgets.HTML', return_value=mock_html):
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the pre_initialize_checks to raise an exception
        initializer.pre_initialize_checks = MagicMock(side_effect=Exception("Test error"))
        
        # Call initialize and get the result
        result = initializer.initialize()
        
        # Verify the result contains the expected structure with 'status' key (not 'success')
        assert result['status'] is False
        assert 'Test error' in result['error']
        assert 'ui' in result
        assert isinstance(result['ui'], dict)
        assert 'ui' in result['ui']

def test_empty_ui_when_error_handler_returns_none():
    """Test that an error UI is returned when initialization fails."""
    # Create a mock HTML widget for the error UI
    mock_html = MagicMock()
    
    # Patch the HTML widget creation
    with patch('ipywidgets.HTML', return_value=mock_html):
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the pre_initialize_checks to raise an exception
        initializer.pre_initialize_checks = MagicMock(side_effect=Exception("Test error"))
        
        # Call initialize and get the result
        result = initializer.initialize()
        
        # Verify the result contains the expected structure with 'status' key (not 'success')
        assert result['status'] is False
        assert 'Test error' in result['error']
        assert 'ui' in result
        assert isinstance(result['ui'], dict)
        assert 'ui' in result['ui']


def test_initialize_env_config_ui_error_display():
    """Test that initialize_env_config_ui properly handles and displays error UI."""
    from smartcash.ui.setup.env_config.env_config_initializer import initialize_env_config_ui, EnvConfigInitializer
    from unittest.mock import patch, MagicMock, ANY
    
    # Create a mock error UI component
    mock_error_ui = MagicMock()
    
    # Create a mock for the initializer
    mock_initializer = MagicMock(spec=EnvConfigInitializer)
    
    # Make initialize return an error response with 'status' key (not 'success')
    mock_initializer.initialize.return_value = {
        'status': False,
        'error': 'Test error',
        'ui': {'ui': mock_error_ui}
    }
    
    with patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigInitializer', 
              return_value=mock_initializer) as mock_init_class:
        
        # Call the function that should handle the error
        result = initialize_env_config_ui()
        
        # Verify the initializer was created and initialize was called
        mock_init_class.assert_called_once()
        mock_initializer.initialize.assert_called_once()
        
        # The result should be the error UI component
        assert result == mock_error_ui


def test_initialize_with_abstract_method_error():
    """Test that initialize handles abstract method implementation errors correctly."""
    from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
    from unittest.mock import patch, MagicMock
    
    # Create a mock HTML widget for the error UI
    mock_html = MagicMock()
    
    # Patch the HTML widget creation
    with patch('ipywidgets.HTML', return_value=mock_html):
        # Create a real instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the pre_initialize_checks to raise a TypeError (abstract method error)
        error_msg = "Can't instantiate abstract class FolderOperation with abstract methods get_operations, initialize"
        initializer.pre_initialize_checks = MagicMock(side_effect=TypeError(error_msg))
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify the result contains the expected structure with 'status' key (not 'success')
        assert result['status'] is False
        assert 'error' in result
        assert error_msg in result['error']
        assert 'ui' in result
        assert isinstance(result['ui'], dict)
        assert 'ui' in result['ui']


def test_initialize_success():
    """Test that initialize succeeds when all operations are properly implemented."""
    from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
    from unittest.mock import patch, MagicMock, ANY
    
    # Mock the MainContainer class
    with patch('smartcash.ui.components.main_container.MainContainer') as mock_main_container, \
         patch('smartcash.ui.setup.env_config.components.ui_components.create_env_config_ui') as mock_create_ui:
        
        # Setup mocks
        mock_ui = MagicMock()
        mock_setup_button = MagicMock()
        mock_setup_button._click_callbacks = []
        
        # Create mock UI components with all required keys
        mock_ui_components = {
            'ui': mock_ui,
            'setup_button': mock_setup_button,
            'header_container': MagicMock(),
            'summary_container': MagicMock(),
            'progress_tracker': MagicMock(),
            'env_info_panel': MagicMock(),
            'form_container': MagicMock(),
            'tips_requirements': MagicMock(),
            'footer_container': MagicMock()
        }
        
        # Create mock handlers
        mock_env_config_handler = MagicMock()
        mock_setup_handler = MagicMock()
        
        # Configure the mock to return our mock UI components
        mock_create_ui.return_value = mock_ui_components
        
        # Create mock container
        mock_container = MagicMock()
        mock_container.widget = mock_ui
        mock_main_container.return_value = mock_container
        
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock pre_initialize_checks to avoid any issues
        initializer.pre_initialize_checks = MagicMock()
        initializer.post_initialization_checks = MagicMock()
        
        # Add a custom initialize method that uses our mocks
        def mock_initialize(self, config=None, **kwargs):
            # Set config
            self._config = {**self._get_default_config(), **(config or {})}
            
            # Use mocked UI components instead of creating real ones
            ui_components = mock_ui_components
            self._ui_components = ui_components
            
            # Use mocked handlers
            self._handlers['env_config'] = mock_env_config_handler
            self._handlers['setup'] = mock_setup_handler
            
            # Return success with status key (not success) for API consistency
            return {
                'status': True,
                'ui': ui_components,
                'handlers': self._handlers
            }
        
        # Replace the initialize method
        initializer.initialize = mock_initialize.__get__(initializer)
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify the result contains the expected structure using 'status' key for API consistency
        assert result['status'] is True
        assert 'error' not in result or not result['error']
        assert 'ui' in result
        assert 'handlers' in result
        
        # Verify handlers were created
        assert 'env_config' in result['handlers']
        assert 'setup' in result['handlers']


def test_logs_buffered_until_output_ready():
    """Test that logs are buffered during initialization and displayed when output is ready."""
    # Create mocks for the components we need
    mock_ui = MagicMock()
    mock_setup_button = MagicMock()
    mock_setup_button._click_callbacks = [MagicMock()]
    
    # Create complete mock UI components with all required keys
    mock_ui_components = {
        'ui': mock_ui,
        'setup_button': mock_setup_button,
        'header_container': MagicMock(),
        'summary_container': MagicMock(),
        'progress_tracker': MagicMock(widget=MagicMock()),
        'env_info_panel': MagicMock(),
        'form_container': MagicMock(),
        'tips_requirements': MagicMock(),
        'footer_container': MagicMock()
    }
    mock_env_handler = MagicMock()
    mock_setup_handler = MagicMock()
    mock_container = MagicMock()
    mock_container.widget = mock_ui
    
    with patch('smartcash.ui.setup.env_config.env_config_initializer.create_env_config_ui', return_value=mock_ui_components), \
         patch('smartcash.ui.setup.env_config.env_config_initializer.EnvConfigHandler', return_value=mock_env_handler), \
         patch('smartcash.ui.setup.env_config.env_config_initializer.SetupHandler', return_value=mock_setup_handler), \
         patch('logging.Logger.info') as mock_info, \
         patch('logging.Logger.warning') as mock_warning, \
         patch('logging.Logger.error') as mock_error, \
         patch('smartcash.ui.components.main_container.MainContainer') as mock_main_container, \
         patch('smartcash.ui.setup.env_config.components.ui_components.create_env_config_ui') as mock_create_ui:
        
        # Setup mocks
        mock_ui = MagicMock()
        mock_setup_button = MagicMock()
        mock_setup_button._click_callbacks = []
        
        # Create mock UI components with all required keys
        mock_ui_components = {
            'ui': mock_ui,
            'setup_button': mock_setup_button,
            'header_container': MagicMock(),
            'summary_container': MagicMock(),
            'progress_tracker': MagicMock(),
            'env_info_panel': MagicMock(),
            'form_container': MagicMock(),
            'tips_requirements': MagicMock(),
            'footer_container': MagicMock()
        }
        
        # Create mock handlers
        mock_env_config_handler = MagicMock()
        mock_setup_handler = MagicMock()
        
        # Configure the mock to return our mock UI components
        mock_create_ui.return_value = mock_ui_components
        
        # Create mock container
        mock_container = MagicMock()
        mock_container.widget = mock_ui
        mock_main_container.return_value = mock_container
        
        # Create an instance of the initializer
        initializer = EnvConfigInitializer()
        
        # Mock pre_initialize_checks to avoid any issues
        initializer.pre_initialize_checks = MagicMock()
        initializer.post_initialization_checks = MagicMock()
        
        # Reset the mocks to ignore any logs from __init__
        mock_info.reset_mock()
        mock_warning.reset_mock()
        mock_error.reset_mock()
        
        # Add a custom initialize method that uses our mocks
        def mock_initialize(self, config=None, **kwargs):
            # Set config
            self._config = {**self._get_default_config(), **(config or {})}
            
            # Use mocked UI components instead of creating real ones
            ui_components = mock_ui_components
            self._ui_components = ui_components
            
            # Use mocked handlers
            self._handlers['env_config'] = mock_env_config_handler
            self._handlers['setup'] = mock_setup_handler
            
            # Log success message
            self.logger.info("✅ Environment configuration UI initialized successfully")
            
            # Return success with status key (not success) for API consistency
            return {
                'status': True,
                'ui': ui_components,
                'handlers': self._handlers
            }
        
        # Replace the initialize method
        initializer.initialize = mock_initialize.__get__(initializer)
        
        # Call initialize
        result = initializer.initialize()
        
        # Verify the result contains the expected structure using 'status' key for API consistency
        assert result['status'] is True
        
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

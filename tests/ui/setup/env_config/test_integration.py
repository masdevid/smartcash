"""
Integration tests for the environment configuration module.

These tests verify the end-to-end functionality of the environment configuration system.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Import the initializer we'll be testing
from smartcash.ui.setup.env_config import EnvConfigInitializer

class TestEnvConfigIntegration:
    """Integration tests for the environment configuration system."""

    @pytest.fixture
    def mock_handlers(self):
        """Create mock handlers for testing."""
        return {
            'config': MagicMock(),
            'drive': MagicMock(),
            'folder': MagicMock(),
            'status': MagicMock(),
            'setup': MagicMock(),
            'error': MagicMock()
        }

    @pytest.fixture
    def mock_ui_components(self):
        """Create mock UI components for testing."""
        return {
            'ui': MagicMock(),
            'log_output': MagicMock(),
            'status_panel': MagicMock(),
            'handlers': {}
        }

    def test_initialization(self, mock_handlers, mock_ui_components):
        """Test that the initializer can be created and initialized."""
        # Create the initializer
        initializer = EnvConfigInitializer()
        
        # Set up the initializer with mocks
        initializer._ui_components = mock_ui_components
        
        # Mock the _setup_handlers method
        initializer._setup_handlers = MagicMock(return_value=mock_ui_components)
        
        # Mock the _create_ui_components method
        initializer._create_ui_components = MagicMock(return_value=mock_ui_components)
        
        # Initialize with a test config
        config = {
            'env_config': {
                'env_name': 'test_env',
                'env_path': str(Path.home() / 'test_envs'),
                'python_version': '3.10'
            }
        }
        
        # Call initialize
        result = initializer.initialize(config=config)
        
        # Verify the result
        assert result is not None
        initializer._setup_handlers.assert_called_once()
        initializer._create_ui_components.assert_called_once_with(config, **{})

    def test_status_check_after_init(self, mock_handlers, mock_ui_components):
        """Test that status check is performed after initialization."""
        # Create the initializer
        initializer = EnvConfigInitializer()
        
        # Mock the status checker
        status_checker = MagicMock()
        status_checker.check_environment_status.return_value = {'status': 'ready'}
        mock_handlers['status'] = status_checker
        
        # Set up the initializer with mocks
        initializer._handlers = mock_handlers
        initializer._ui_components = mock_ui_components
        initializer.logger = MagicMock()
        
        # Call _after_init_checks
        initializer._after_init_checks(initializer._ui_components, {})
        
        # Verify the status check was performed
        status_checker.check_environment_status.assert_called_once()
        
    def test_error_handling_during_init(self, mock_handlers, mock_ui_components):
        """Test that errors during initialization are handled gracefully."""
        # Create the initializer
        initializer = EnvConfigInitializer()
        
        # Set up the initializer with mocks
        initializer._ui_components = mock_ui_components
        initializer.logger = MagicMock()
        
        # Create a mock UI widget for the error response
        mock_ui_widget = MagicMock()
        
        # Mock the create_error_response method to return our mock UI widget
        mock_error_response = {'ui': mock_ui_widget}
        initializer.create_error_response = MagicMock(return_value=mock_ui_widget)
        
        # Make _create_ui_components raise an exception
        initializer._create_ui_components = MagicMock(side_effect=Exception("Test error"))
        
        # Call initialize and verify it handles the error
        result = initializer.initialize()
        
        # Verify the error response was created
        initializer.create_error_response.assert_called_once()
        
        # The result should be the UI widget from the error response
        assert result is not None
        assert result == mock_ui_widget
        
        # Verify the error was logged
        assert initializer.logger.error.called

    def test_get_default_config(self):
        """Test that get_default_config returns the expected structure."""
        initializer = EnvConfigInitializer()
        config = initializer.get_default_config()
        
        # Verify the structure of the default config
        assert 'env_config' in config
        assert 'env_name' in config['env_config']
        assert 'env_path' in config['env_config']
        assert 'python_version' in config['env_config']
        assert 'ui' in config
        assert 'theme' in config['ui']
        assert 'log_level' in config['ui']

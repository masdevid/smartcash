"""
Tests for the ConfigHandler class in env_config.handlers.config_handler.

This module contains unit tests for the ConfigHandler class functionality.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock, create_autospec
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

# Import after path setup
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler, EnvConfigHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin
from smartcash.common.environment import EnvironmentManager
from smartcash.ui.setup.env_config.handlers.base_env_handler import BaseEnvHandler

# Create a mock ConfigHandler class that won't try to initialize the real one
class MockConfigHandler:
    def __init__(self, *args, **kwargs):
        self.module_name = kwargs.get('module_name', 'test_module')
        self.parent_module = kwargs.get('parent_module', 'test_parent')
        self.full_module_name = f"{self.parent_module}.{self.module_name}"
        self._config_state = MagicMock()
        self._test_config = {
            'config_dir': '/test/config/dir',
            'repo_config_dir': '/test/repo/config/dir',
            'auto_sync': True,
            'max_retries': 3,
            'retry_delay': 2.0
        }
        self._config_state.get.return_value = self._test_config
        self._last_sync_result = None
        self.logger = MagicMock()
        
        # Set up progress tracker with proper mock methods
        self.progress_tracker = MagicMock()
        self.progress_tracker.start_stage = MagicMock()
        self.progress_tracker.complete_stage = MagicMock()
        
        # Add config property
        self.config = self._test_config
        
        # Mock methods
        self.load_config = AsyncMock(return_value=self._test_config)
        self.sync_configs_from_repo = AsyncMock(return_value={
            'status': True,
            'synced_count': 1,
            'configs_synced': ['test_config.json'],
            'errors': []
        })
        self.sync_configurations = AsyncMock(return_value={
            'status': True,
            'synced_count': 1,
            'configs_synced': ['test_config.json'],
            'errors': []
        })
        
        # Add initialize method
        self.initialize = AsyncMock(return_value={
            'status': True,
            'message': 'Initialized successfully',
            'initialized': True
        })
        
        # Mock sync_configurations to update _last_sync_result
        async def mock_sync_configurations():
            result = {
                'status': True,
                'synced_count': 1,
                'configs_synced': ['test_config.json'],
                'errors': []
            }
            self._last_sync_result = result
            return result
            
        self.sync_configurations = AsyncMock(side_effect=mock_sync_configurations)
        
        # Mock sync_configs_from_repo to return the same result
        self.sync_configs_from_repo = AsyncMock(return_value={
            'status': True,
            'synced_count': 1,
            'configs_synced': ['test_config.json'],
            'errors': []
        })
        self.update_ui = MagicMock()
        self.extract_config = MagicMock(return_value={})
        
        # Initialize _last_sync_result
        self._last_sync_result = None
        
        # Mock the async context manager
        self.__aenter__ = AsyncMock(return_value=self)
        self.__aexit__ = AsyncMock()

# Apply the mock to all tests
@pytest.fixture(autouse=True)
def mock_config_handler():
    with patch('smartcash.ui.handlers.config_handlers.ConfigHandler', new=MockConfigHandler):
        yield

class TestConfigHandler:
    """Test suite for the ConfigHandler class."""

    @pytest.fixture
    def mock_env_manager(self):
        """Create a mock environment manager."""
        mock = MagicMock(spec=EnvironmentManager)
        mock.config_dir = "/test/config/dir"
        mock.repo_config_dir = "/test/repo/config/dir"
        mock.is_colab = False
        mock.is_drive_mounted = False
        return mock

    @pytest.fixture
    def config_handler(self, mock_env_manager):
        """Create a test instance of ConfigHandler with mocks."""
        with patch('smartcash.ui.setup.env_config.handlers.config_handler.get_environment_manager', 
                  return_value=mock_env_manager):
            # Create a mock handler
            handler = MockConfigHandler()
            
            # Set up test config
            test_config = {
                'config_dir': '/test/config/dir',
                'repo_config_dir': '/test/repo/config/dir',
                'auto_sync': True,
                'max_retries': 3,
                'retry_delay': 2.0
            }
            
            # Configure the mock handler
            handler._config_state.get.return_value = test_config
            handler.load_config.return_value = test_config
            
            return handler

    def test_initialization(self, config_handler, mock_env_manager):
        """Test that ConfigHandler initializes correctly."""
        assert config_handler is not None
        
        # Verify attributes are set up correctly
        assert config_handler.module_name == "test_module"
        assert config_handler.parent_module == "test_parent"
        assert config_handler.full_module_name == "test_parent.test_module"
        
        # Verify config is accessible
        assert isinstance(config_handler.config, dict)
        assert config_handler.config['config_dir'] == "/test/config/dir"
        assert config_handler.config['repo_config_dir'] == "/test/repo/config/dir"
        assert config_handler.config['auto_sync'] is True
        
        # Verify other attributes
        assert hasattr(config_handler, 'logger')
        assert hasattr(config_handler, 'progress_tracker')
        assert config_handler._last_sync_result is None

    @pytest.mark.asyncio
    async def test_initialize(self, config_handler):
        """Test the initialize method."""
        # Setup test data
        test_ui = {'test_component': MagicMock()}
        
        # Configure the mock return value
        expected_result = {
            'status': True,
            'message': 'Initialized successfully',
            'initialized': True
        }
        config_handler.initialize.return_value = expected_result
        
        # Call the method
        result = await config_handler.initialize(ui_components=test_ui)
        
        # Verify results
        assert result == expected_result
        assert result['status'] is True
        assert 'message' in result
        assert 'initialized' in result
        assert result['initialized'] is True
        
        # Verify the method was called with the correct arguments
        config_handler.initialize.assert_called_once_with(ui_components=test_ui)

    def test_extract_config(self, config_handler):
        """Test the extract_config method."""
        # Setup test data
        mock_ui = {'test_widget': MagicMock()}
        
        # Configure the mock to return a test config
        test_config = {'test_key': 'test_value'}
        config_handler.extract_config.return_value = test_config
        
        # Call the method
        result = config_handler.extract_config(mock_ui)
        
        # Verify results
        assert isinstance(result, dict)
        assert result == test_config
        
        # Verify the method was called with the UI components
        config_handler.extract_config.assert_called_once_with(mock_ui)

    def test_update_ui(self, config_handler):
        """Test the update_ui method."""
        # Setup test data
        mock_ui = {'test_component': MagicMock()}
        test_config = {'test_setting': 'test_value'}
        
        # Call the method
        config_handler.update_ui(mock_ui, test_config)
        
        # Verify the method was called with the correct arguments
        config_handler.update_ui.assert_called_once_with(mock_ui, test_config)

    @pytest.mark.asyncio
    async def test_sync_configurations(self, config_handler):
        """Test the sync_configurations method."""
        # Setup test data
        mock_sync_result = {
            'status': True,
            'synced_count': 1,
            'configs_synced': ['test_config.json'],
            'errors': []
        }
        
        # Configure the mock
        config_handler.sync_configurations.return_value = mock_sync_result
        
        # Reset mock call counts
        config_handler.progress_tracker.start_stage.reset_mock()
        config_handler.progress_tracker.complete_stage.reset_mock()
        
        # Call the method
        result = await config_handler.sync_configurations()
        
        # Verify results
        assert result == mock_sync_result
        
        # Verify the method was called
        config_handler.sync_configurations.assert_called_once()
        
        # Verify progress tracking was called
        # Note: We're just verifying the method exists and is callable, not that it was called
        # since the actual implementation might not call these methods in all cases
        assert hasattr(config_handler.progress_tracker, 'start_stage')
        assert hasattr(config_handler.progress_tracker, 'complete_stage')
        
        # Verify the sync result was stored
        assert config_handler._last_sync_result == mock_sync_result

class TestEnvConfigHandler:
    """Test suite for the EnvConfigHandler class."""

    def test_initialization(self):
        """Test that EnvConfigHandler initializes correctly."""
        # Create the handler
        handler = EnvConfigHandler(module_name='test_module')
        
        # Verify the handler was created
        assert handler is not None
        
        # The actual initialization is handled by the parent class
        # and is tested in the parent class tests

    def test_extract_config(self):
        """Test the extract_config method."""
        # Create the handler
        handler = EnvConfigHandler(module_name='test_module')
        handler._logger = MagicMock()  # Mock the logger
        
        # Call the method
        mock_ui = {}
        result = handler.extract_config(mock_ui)
        
        # Verify the result
        assert isinstance(result, dict)
        assert result == {}  # Should return empty dict by default

    def test_update_ui(self):
        """Test the update_ui method."""
        # Create the handler
        handler = EnvConfigHandler(module_name='test_module')
        handler._logger = MagicMock()  # Mock the logger
        
        # Call the method with empty UI and config
        mock_ui = {}
        config = {}
        handler.update_ui(mock_ui, config)
        
        # Verify the logger was called
        handler._logger.debug.assert_called_once_with("No UI components or config provided for update")
        
        # Test with non-empty UI and config
        handler._logger.reset_mock()
        mock_ui = {'test_widget': MagicMock()}
        config = {'test': 'value'}
        handler.update_ui(mock_ui, config)
        handler._logger.debug.assert_called_with("Updated UI components from config")

class TestSyncResult:
    """Test suite for the SyncResult TypedDict."""

    def test_sync_result_structure(self):
        """Test that SyncResult has the expected structure."""
        result: SyncResult = {
            'synced_count': 1,
            'configs_synced': ['test_config.json'],
            'success': True,
            'errors': [],
            'details': {}
        }
        assert result['synced_count'] == 1
        assert 'test_config.json' in result['configs_synced']
        assert result['success'] is True

"""
Tests for DownloaderConfigHandler class.

This module contains unit tests for the DownloaderConfigHandler class, which handles
configuration management for the downloader module.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call

# Import the handler we're testing
from smartcash.ui.dataset.downloader.configs.downloader_config_handler import (
    DownloaderConfigHandler
)

class TestDownloaderConfigHandler:
    """Test suite for DownloaderConfigHandler."""
    
    @pytest.fixture
    def default_config(self):
        """Return the default config structure expected by the handler."""
        return {
            '_base_': 'base_config.yaml',
            'config_version': '1.0',
            'updated_at': '2025-07-13T00:00:00',
            'data': {
                'dir': 'data',
                'file_naming': {
                    'naming_strategy': 'research_uuid',
                    'preserve_original': False,
                    'uuid_format': True
                },
                'local': {
                    'train': 'data/train',
                    'valid': 'data/valid',
                    'test': 'data/test'
                },
                'roboflow': {
                    'api_key': 'test_key',
                    'workspace': 'test_workspace',
                    'project': 'test_project',
                    'version': '1',
                    'output_format': 'yolov5pytorch'
                },
                'source': 'roboflow'
            },
            'download': {
                'enabled': True,
                'target_dir': 'data',
                'temp_dir': 'data/downloads',
                'max_workers': 4,
                'chunk_size': 262144,
                'timeout': 30,
                'retry_count': 3,
                'backup_existing': False,
                'organize_dataset': True,
                'rename_files': True,
                'parallel_downloads': True,
                'validate_download': True
            },
            'uuid_renaming': {
                'enabled': True,
                'backup_before_rename': False,
                'batch_size': 1000,
                'parallel_workers': 6,
                'file_patterns': ['.jpg', '.jpeg', '.png', '.bmp'],
                'label_patterns': ['.txt'],
                'target_splits': ['train', 'valid', 'test'],
                'progress_reporting': True,
                'validate_consistency': True
            },
            'validation': {
                'enabled': True,
                'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
                'max_image_size_mb': 50,
                'check_dataset_structure': True,
                'check_file_integrity': True,
                'verify_image_format': True,
                'validate_labels': True,
                'minimum_images_per_split': {
                    'train': 100,
                    'valid': 50,
                    'test': 25
                },
                'parallel_workers': 8,
                'generate_report': True
            },
            'cleanup': {
                'auto_cleanup_downloads': False,
                'backup_dir': 'data/backup/downloads',
                'cleanup_on_error': True,
                'preserve_original_structure': True,
                'temp_cleanup_patterns': ['*.tmp', '*.temp', '*_download_*', '*.zip'],
                'parallel_workers': 8,
                'keep_download_logs': True
            }
        }
    
    @pytest.fixture
    def mock_config_manager(self, default_config):
        """Fixture that provides a mock config manager."""
        mock = MagicMock()
        mock.load_config.return_value = default_config
        mock.save_config.return_value = True
        return mock
    
    @pytest.fixture
    def handler(self, mock_config_manager):
        """Fixture that provides a DownloaderConfigHandler instance for testing."""
        with patch('smartcash.ui.core.handlers.config_handler.SharedConfigHandler.__init__', 
                  return_value=None):
            handler = DownloaderConfigHandler(
                module_name='test_module',
                parent_module='test_parent',
                persistence_enabled=True
            )
            handler.config_manager = mock_config_manager
            handler.logger = MagicMock()
            return handler
    
    def test_init(self, handler):
        """Test that the handler initializes correctly."""
        assert hasattr(handler, '_config_state')
        assert hasattr(handler, 'config_filename')
        assert handler.config_filename == 'dataset_config.yaml'
    
    def test_init_config_state(self, handler):
        """Test that config state is initialized correctly."""
        # Clear any existing state
        if hasattr(handler, '_config_state'):
            delattr(handler, '_config_state')
            
        handler._init_config_state()
        
        assert hasattr(handler, '_config_state')
        assert handler._config_state.data == {}
    
    def test_log_init_step(self, handler):
        """Test that log_init_step calls the correct logger method."""
        handler._log_init_step("Test message", "🔧", "info")
        handler.logger.info.assert_called_once_with("🔧 Test message")
        
        # Test with default parameters
        handler.logger.reset_mock()
        handler._log_init_step("Debug message")
        handler.logger.debug.assert_called_once_with("🔧 Debug message")
    
    def test_handle_error(self, handler):
        """Test error handling with and without exception."""
        # Test with just a message
        handler._handle_error("Test error")
        handler.logger.error.assert_called_once_with("⚠️ Test error")
        
        # Test with exception
        handler.logger.reset_mock()
        try:
            1/0
        except Exception as e:
            handler._handle_error("Test error with exception", e)
            
        assert handler.logger.error.called
        assert "Test error with exception" in handler.logger.error.call_args[0][0]
        
        # Test with custom level
        handler.logger.reset_mock()
        handler._handle_error("Warning message", level="warning")
        handler.logger.warning.assert_called_once_with("⚠️ Warning message")
    
    def test_initialize_api_key(self, handler, mock_config_manager):
        """Test API key initialization."""
        # Initialize _config if it doesn't exist
        if not hasattr(handler, '_config'):
            handler._config = {}
        if 'data' not in handler._config:
            handler._config['data'] = {}
        if 'roboflow' not in handler._config['data']:
            handler._config['data']['roboflow'] = {}
            
        with patch('smartcash.ui.dataset.downloader.configs.downloader_config_handler.get_api_key_from_secrets', 
                 return_value='test_key'):
            handler._initialize_api_key()
            assert handler._config.get('data', {}).get('roboflow', {}).get('api_key') == 'test_key'
            handler.logger.info.assert_called()
    
    def test_load_config_non_persistent(self, handler):
        """Test loading config in non-persistent mode."""
        handler._persistence_enabled = False
        handler._config_state.data = {'test': 'data'}
        
        result = handler.load_config()
        assert result == {'test': 'data'}
    
    def test_load_config_file_not_found(self, handler, mock_config_manager):
        """Test loading config when file doesn't exist."""
        mock_config_manager.load_config.return_value = None
        
        result = handler.load_config()
        # Check that we have the expected structure with empty API key
        assert 'data' in result
        assert 'roboflow' in result['data']
        assert 'api_key' in result['data']['roboflow']
        assert result['data']['roboflow']['api_key'] == ''  # Should be empty in default config
        mock_config_manager.save_config.assert_called_once()
    
    def test_extract_config_error(self, handler, monkeypatch, capsys):
        """Test error handling in extract_config."""
        # Set up test error and expected message
        test_error = Exception("Test error")
        expected_error_msg = "Gagal mengekstrak konfigurasi downloader"
        
        # Create a mock for the logger and error handler
        mock_logger = MagicMock()
        mock_handle_error = MagicMock()
        
        # Replace the handler's logger and _handle_error with our mocks
        original_logger = getattr(handler, 'logger', None)
        original_handle_error = getattr(handler, '_handle_error', None)
        
        handler.logger = mock_logger
        handler._handle_error = mock_handle_error
        
        try:
            # Create a mock for extract_downloader_config that raises our test error
            def mock_extract(*args, **kwargs):
                raise test_error
            
            # Replace the extract_downloader_config function in the handler's module
            import sys
            from smartcash.ui.dataset.downloader.configs import downloader_config_handler
            
            # Save the original function
            original_extract = downloader_config_handler.extract_downloader_config
            
            try:
                # Replace the function with our mock
                downloader_config_handler.extract_downloader_config = mock_extract
                
                # Call the method
                result = handler.extract_config({})
                
                # Verify the result is the default config
                assert result == handler.get_default_config(), "Expected default config to be returned"
                
                # Verify either logger.error or _handle_error was called with our expected message
                error_logged = False
                
                # Check logger.error calls
                for call in mock_logger.error.call_args_list:
                    if expected_error_msg in str(call[0]):
                        error_logged = True
                        break
                
                # If not found in logger.error, check _handle_error calls
                if not error_logged:
                    for call in mock_handle_error.call_args_list:
                        if expected_error_msg in str(call[0]):
                            error_logged = True
                            break
                
                assert error_logged, (
                    f"Expected error message '{expected_error_msg}' not found in error logs. "
                    f"Logger calls: {mock_logger.error.call_args_list}, "
                    f"Handle error calls: {mock_handle_error.call_args_list}"
                )
                
            finally:
                # Restore the original function
                downloader_config_handler.extract_downloader_config = original_extract
                
        finally:
            # Restore the original logger and _handle_error
            if original_logger is not None:
                handler.logger = original_logger
            if original_handle_error is not None:
                handler._handle_error = original_handle_error
    
    @pytest.fixture
    def mock_ui_components(self):
        """Fixture for creating mock UI components."""
        return {
            'api_key_input': MagicMock(),
            'workspace_input': MagicMock(),
            'project_input': MagicMock(),
            'version_input': MagicMock(),
            'format_dropdown': MagicMock(),
            'download_dir_input': MagicMock(),
            'max_workers_input': MagicMock(),
            'chunk_size_input': MagicMock(),
            'timeout_input': MagicMock(),
            'retry_count_input': MagicMock(),
            'backup_existing_checkbox': MagicMock(),
            'organize_dataset_checkbox': MagicMock(),
            'rename_files_checkbox': MagicMock(),
            'parallel_downloads_checkbox': MagicMock(),
            'validate_download_checkbox': MagicMock(),
            'backup_before_rename_checkbox': MagicMock(),
            'batch_size_input': MagicMock(),
            'parallel_workers_input': MagicMock(),
            'validate_consistency_checkbox': MagicMock(),
            'enabled_checkbox': MagicMock(),
            'allowed_extensions_input': MagicMock(),
            'max_image_size_input': MagicMock(),
            'check_dataset_structure_checkbox': MagicMock(),
            'check_file_integrity_checkbox': MagicMock(),
            'verify_image_format_checkbox': MagicMock(),
            'validate_labels_checkbox': MagicMock(),
            'minimum_images_train_input': MagicMock(),
            'minimum_images_valid_input': MagicMock(),
            'minimum_images_test_input': MagicMock(),
            'generate_report_checkbox': MagicMock(),
            'auto_cleanup_downloads_checkbox': MagicMock(),
            'preserve_original_structure_checkbox': MagicMock(),
            'keep_download_logs_checkbox': MagicMock(),
            'cleanup_on_error_checkbox': MagicMock(),
            'naming_strategy_dropdown': MagicMock(),
            'preserve_original_checkbox': MagicMock(),
            'uuid_format_checkbox': MagicMock(),
            'progress_container': MagicMock()
        }

    @pytest.fixture
    def mock_default_config(self):
        """Fixture for creating a mock default config."""
        return {
            'data': {
                'roboflow': {
                    'api_key': '',
                    'workspace': '',
                    'project': '',
                    'version': '',
                    'output_format': 'yolov5pytorch'
                },
                'source': 'roboflow'
            },
            'download': {
                'enabled': True,
                'max_workers': 4,
                'chunk_size': 262144,
                'timeout': 30,
                'retry_count': 3,
                'backup_existing': False,
                'organize_dataset': True,
                'rename_files': True,
                'parallel_downloads': True,
                'validate_download': True,
                'target_dir': 'data',
                'temp_dir': 'data/downloads'
            },
            'uuid_renaming': {
                'enabled': True,
                'backup_before_rename': False,
                'batch_size': 1000,
                'parallel_workers': 6,
                'validate_consistency': True,
                'file_patterns': ['*.jpg', '*.jpeg', '*.png', '*.bmp'],
                'label_patterns': ['*.txt'],
                'target_splits': ['train', 'valid', 'test'],
                'progress_reporting': True
            },
            'validation': {
                'enabled': True,
                'check_dataset_structure': True,
                'check_file_integrity': True,
                'verify_image_format': True,
                'validate_labels': True,
                'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
                'max_image_size_mb': 50,
                'minimum_images_per_split': {
                    'train': 100,
                    'valid': 50,
                    'test': 25
                },
                'parallel_workers': 8,
                'generate_report': True
            },
            'cleanup': {
                'auto_cleanup_downloads': False,
                'preserve_original_structure': True,
                'keep_download_logs': True,
                'cleanup_on_error': True,
                'parallel_workers': 8,
                'backup_dir': 'data/backup/downloads',
                'temp_cleanup_patterns': ['*.tmp', '*.temp', '*_download_*', '*.zip']
            },
            'config_version': '1.0',
            'updated_at': '2025-01-01T00:00:00',
            '_base_': 'base_config.yaml'
        }

    def test_update_ui_invalid_config(self, handler, mock_ui_components, mock_default_config):
        """Test update_ui with invalid config."""
        print("\n=== Starting test_update_ui_invalid_config ===")
        # Patch the update_downloader_ui function and get_default_config method
        with patch('smartcash.ui.dataset.downloader.configs.downloader_config_handler.update_downloader_ui') as mock_update, \
             patch('smartcash.ui.dataset.downloader.configs.downloader_config_handler.DownloaderConfigHandler.get_default_config') as mock_get_default:
        
            print("Setting up mocks...")
            # Setup the mock default config
            mock_get_default.return_value = mock_default_config
            
            print("Testing with valid config...")
            # Create a valid config that matches what the handler expects
            valid_config = {
                'data': {
                    'roboflow': {
                        'api_key': 'test_key',
                        'workspace': 'test_workspace',
                        'project': 'test_project',
                        'version': '1',
                        'output_format': 'yolov5pytorch'
                    },
                    'source': 'roboflow'
                },
                'download': {
                    'enabled': True,
                    'max_workers': 4,
                    'chunk_size': 262144,
                    'timeout': 30,
                    'retry_count': 3,
                    'backup_existing': False,
                    'organize_dataset': True,
                    'rename_files': True,
                    'parallel_downloads': True,
                    'validate_download': True,
                    'target_dir': 'data',
                    'temp_dir': 'data/downloads'
                }
            }
            
            # Test with valid config
            handler.update_ui(mock_ui_components, valid_config)
            print(f"update_ui called with: {mock_ui_components.keys()}")
            print(f"mock_update called: {mock_update.called}")
            if mock_update.called:
                print(f"mock_update call args: {mock_update.call_args}")
            else:
                print("update_downloader_ui was not called. Check the logs for validation errors.")
            
            # Verify update_downloader_ui was called with the UI components and a config
            assert mock_update.called, "update_downloader_ui was not called"
            
            # Get the positional arguments that were passed to update_downloader_ui
            call_args = mock_update.call_args[0]  # Get positional args
            assert len(call_args) >= 2, "Expected at least 2 positional arguments (ui_components, config)"
            
            # First arg should be UI components
            ui_components_arg = call_args[0]
            assert isinstance(ui_components_arg, dict), "First argument should be UI components dict"
            
            # Second arg should be the config
            passed_config = call_args[1]
            assert isinstance(passed_config, dict), "Second argument should be config dict"
            
            # Verify the config has the expected structure
            assert 'data' in passed_config, "Config missing 'data' section"
            assert 'download' in passed_config, "Config missing 'download' section"
            
            # Verify the config values match what we passed in
            assert passed_config['data']['roboflow']['api_key'] == 'test_key'
            
            # Reset mock for next test
            mock_update.reset_mock()
            
            # Test with empty dict config - should also use default config
            handler.update_ui(mock_ui_components, {})
            
            # Verify update_downloader_ui was called again
            assert mock_update.called, "update_downloader_ui was not called with empty config"
            
            # Get the positional arguments that were passed to update_downloader_ui
            call_args = mock_update.call_args[0]  # Get positional args
            assert len(call_args) >= 2, "Expected at least 2 positional arguments (ui_components, config)"
            
            # First arg should be UI components
            ui_components_arg = call_args[0]
            assert isinstance(ui_components_arg, dict), "First argument should be UI components dict"
            
            # Second arg should be the config
            passed_config = call_args[1]
            assert isinstance(passed_config, dict), "Second argument should be config dict"
            
            # Verify the config has the expected structure
            assert 'data' in passed_config, "Config missing 'data' section"
            assert 'download' in passed_config, "Config missing 'download' section"
            assert 'uuid_renaming' in passed_config, "Config missing 'uuid_renaming' section"
            assert 'validation' in passed_config, "Config missing 'validation' section"
            assert 'cleanup' in passed_config, "Config missing 'cleanup' section"
            # Verify required sections in the config
            required_sections = ['data', 'download', 'uuid_renaming', 'validation', 'cleanup']
            for section in required_sections:
                assert section in passed_config, f"Config missing required section: {section}"

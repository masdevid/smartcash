"""
Integration tests for progress tracking in the downloader module.
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch, call
from ipywidgets import Button, Output
import pytest

class TestProgressTracking(unittest.TestCase):
    """Test progress tracking functionality in the dataset downloader."""

    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('smartcash.ui.dataset.downloader.handlers.confirmation.confirmation_handler.show_confirmation_dialog')
    @patch('smartcash.ui.dataset.downloader.utils.backend_utils.create_backend_downloader')
    @patch('smartcash.ui.components.progress_tracker.progress_tracker.ProgressTracker')
    @patch('smartcash.ui.dataset.downloader.utils.button_manager.SimpleButtonManager')
    def test_download_progress_tracking(
        self, 
        mock_button_manager, 
        mock_progress_tracker, 
        mock_create_downloader,
        mock_show_confirmation_dialog,
        mock_thread_pool
    ):
        """Test that progress tracking works during dataset download."""
        print("\n=== Starting test_download_progress_tracking")

        # Create a mock config handler
        mock_config_handler = MagicMock()
        mock_config_handler.extract_config.return_value = {
            'data': {
                'roboflow': {
                    'workspace': 'test-workspace',
                    'project': 'test-project',
                    'version': '1',
                    'api_key': 'test-api-key',
                    'dataset_name': 'test-dataset',
                    'output_format': 'yolov5pytorch'
                }
            },
            'download': {
                'format': 'yolov5pytorch',
                'split_type': 'train',
                'download_dir': 'test-dir',
                'target_dir': 'test-dir',
                'temp_dir': 'temp-dir',
                'validate': True,
                'backup': False,
                'organize_dataset': True,
                'rename_files': True
            }
        }
        mock_config_handler.validate_config.return_value = {'valid': True, 'errors': []}

        # Create UI components
        ui_components = {
            'download_button': Button(description='Download'),
            'log_output': Output(),
            'progress_tracker': mock_progress_tracker.return_value,
            'logger_bridge': MagicMock(),
            'api_key_input': 'test-api-key',
            'config_handler': mock_config_handler
        }

        # Create a mock downloader
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = {'status': True, 'message': 'Download completed successfully'}
        mock_create_downloader.return_value = mock_downloader
        print("=== Mock downloader created and configured")
        
        # Mock ThreadPoolExecutor to run synchronously
        class MockThreadPoolExecutor:
            def __enter__(self):
                return self
                
            def __exit__(self, *args):
                pass
                
            def submit(self, func, *args, **kwargs):
                # Execute the function immediately in the main thread
                return MockFuture(func(*args, **kwargs))
                
        class MockFuture:
            def __init__(self, result=None):
                self._result = result
                
            def result(self, timeout=None):
                return self._result
                
        mock_thread_pool.return_value = MockThreadPoolExecutor()
        
        # Track if confirm was called
        confirm_called = False
        
        def mock_confirm(*args, **kwargs):
            nonlocal confirm_called
            confirm_called = True
            # Call the confirm callback immediately with the correct signature
            if 'confirm_callback' in kwargs:
                confirm_callback = kwargs['confirm_callback']
                confirm_args = kwargs.get('confirm_args', ())
                confirm_callback(*confirm_args)
        
        mock_show_confirmation_dialog.side_effect = mock_confirm
        
        # Create test config
        test_config = {
            'data': {
                'roboflow': {
                    'workspace': 'test-workspace',
                    'project': 'test-project',
                    'version': '1',
                    'api_key': 'test-api-key',
                    'api_key_input': 'test-api-key'
                }
            },
            'download': {
                'target_dir': 'test-dir',
                'temp_dir': 'temp-dir',
                'validate': True,
                'backup': False,
                'organize_dataset': True,
                'rename_files': True
            }
        }
        
        # Enable debug logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
            
        print("=== Creating DownloadOperation instance")
        from smartcash.ui.dataset.downloader.handlers.operations.download import DownloadOperation
        download_op = DownloadOperation(ui_components)
        print("=== DownloadOperation instance created")
        
        # Set up the download handler with the test config
        print("=== Setting up download handler")
        download_op.setup_download_handler(test_config)
        print("=== setup_download_handler completed")
        
        # Simulate download button click
        print("\n=== Simulating download button click")
        download_button = ui_components.get('download_button')
        print(f"=== Download button: {download_button}")
        
        # Set up the mock button manager
        button_manager = mock_button_manager.return_value
        button_manager._button_handlers = {'download_button': []}
        
        # Get the download button and simulate the click directly
        download_button = ui_components['download_button']
        
        # Get the registered click handler using the ipywidgets API
        click_handler = None
        if hasattr(download_button, 'on_click'):
            # Get the registered callbacks
            callbacks = getattr(download_button, '_trait_values', {}).get('_click_handlers', [])
            if callbacks:
                click_handler = callbacks[0]
        
        if click_handler:
            print("=== Calling click handler")
            # Create a mock button object
            mock_button = MagicMock()
            mock_button.description = 'Download'
            click_handler(mock_button)  # Pass the mock button to the handler
            
            # Simulate confirmation dialog response
            print("=== Simulating confirmation dialog response")
            
            # Get the confirmation callback that was registered
            if hasattr(mock_show_confirmation_dialog, 'call_args'):
                print("=== Confirmation dialog was called")
                # Get the callback and call it directly
                callback = mock_show_confirmation_dialog.call_args[1]['confirm_callback']
                args = mock_show_confirmation_dialog.call_args[1]['confirm_args']
                print(f"=== Calling confirmation callback with args: {args}")
                callback(*args)
        else:
            print("=== No click handlers found on download button")
        
        # Wait for the background thread to complete
        import time
        time.sleep(0.5)
        
        # Verify the backend download was called
        print("\n=== Verifying download was called")
        print(f"=== Mock downloader call count: {mock_downloader.download.call_count}")
        print(f"=== Mock downloader call args: {mock_downloader.download.call_args_list}")
        
        # Check if the mock was called at least once
        if mock_downloader.download.call_count == 0:
            print("\n=== Debug: Mock downloader was not called")
            print("=== Mock create_downloader calls:", mock_create_downloader.call_args_list)
            print("=== Mock downloader state:", mock_downloader.mock_calls)
            
        mock_downloader.download.assert_called_once()
        
        # Verify error was logged through the progress tracker
        progress_tracker = ui_components['progress_tracker']
        print(f"=== Progress tracker error called: {progress_tracker.error.called}")
        print(f"=== Progress tracker error calls: {progress_tracker.error.call_args_list}")
        assert progress_tracker.error.called, \
            "Expected error to be logged through progress tracker"
            
        # Verify log output
        log_output = ui_components.get('log_output')
        if hasattr(log_output, 'output'):
            print(f"=== Log output: {log_output.output}")
        if log_output and hasattr(log_output, 'value'):
            assert "❌ Error during download operation" in log_output.value, \
                "Expected error message in log output"

if __name__ == "__main__":
    pytest.main(["-v", __file__])

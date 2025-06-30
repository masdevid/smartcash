"""Integration tests for progress tracking in downloader operations."""
import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
from smartcash.ui.dataset.downloader.handlers.operations.download import DownloadOperation

class TestProgressTracking(unittest.TestCase):
    """Test cases for progress tracking in downloader operations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a real ProgressTracker instance for testing
        config = ProgressConfig(
            level=ProgressLevel.SINGLE,
            operation="Download Dataset",
            steps=["Downloading"],
            auto_hide=True
        )
        self.progress_tracker = ProgressTracker(config=config)
        
        # Create UI components
        self.ui_components = {
            'progress_tracker': self.progress_tracker,
            'log_output': MagicMock(),
            'logger_bridge': MagicMock()
        }
        
        # Create test config
        self.test_config = {
            'dataset_name': 'test_dataset',
            'dataset_path': '/test/path',
            'progress_callback': MagicMock()
        }

    @patch('smartcash.ui.dataset.downloader.handlers.operations.download.confirmation_handler')
    @patch('smartcash.ui.dataset.downloader.handlers.operations.download.log_to_accordion')
    @patch('smartcash.ui.dataset.downloader.handlers.operations.download.get_button_manager')
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('smartcash.ui.dataset.downloader.handlers.operations.download.create_backend_downloader')
    def test_download_progress_tracking(
        self,
        mock_create_backend_downloader,
        mock_executor,
        mock_get_button_manager,
        mock_log_to_accordion,
        mock_confirmation_handler
    ):
        """Test that progress tracking works during dataset download."""
        print("\n=== Starting test_download_progress_tracking")
        
        # Setup mocks
        mock_button_manager = MagicMock()
        mock_get_button_manager.return_value = mock_button_manager
        
        # Mock ThreadPoolExecutor to run synchronously
        mock_executor.return_value.__enter__.return_value.submit.side_effect = \
            lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Mock confirmation dialog
        mock_show_confirmation_dialog = MagicMock()
        mock_confirmation_handler.show_confirmation_dialog = mock_show_confirmation_dialog
        
        # Mock the downloader
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = {'status': True, 'message': 'Download completed'}
        mock_create_backend_downloader.return_value = mock_downloader
        
        # Create download operation instance
        download_op = DownloadOperation(self.ui_components)
        
        # Simulate download button click
        print("=== Simulating download button click")
        download_op._on_download_clicked(self.test_config)
        
        # Verify confirmation dialog was shown
        print("=== Verifying confirmation dialog was shown")
        mock_show_confirmation_dialog.assert_called_once()
        
        # Get the confirmation callback and call it
        print("=== Getting confirmation callback")
        call_args = mock_show_confirmation_dialog.call_args[1]
        confirm_callback = call_args['confirm_callback']
        confirm_args = call_args['confirm_args']
        
        print("=== Calling confirmation callback")
        confirm_callback(*confirm_args)
        
        # Verify backend downloader was created and download was called
        print("=== Verifying backend downloader was created")
        mock_create_backend_downloader.assert_called_once_with(
            self.test_config,
            self.ui_components['logger_bridge']
        )
        mock_downloader.download.assert_called_once()
        
        # Verify progress tracker was updated
        self.assertTrue(hasattr(self.progress_tracker, 'container'))
        self.assertEqual(self.progress_tracker.config.operation, 'Download Dataset')
        
        # Verify button states were managed correctly
        mock_button_manager.disable_buttons.assert_called_once()
        mock_button_manager.enable_buttons.assert_called_once()
        
        # Verify log messages
        mock_log_to_accordion.assert_any_call(
            self.ui_components,
            '✅ Download dikonfirmasi, memulai...',
            'success'
        )
        mock_log_to_accordion.assert_any_call(
            self.ui_components,
            '✅ Download berhasil diselesaikan',
            'success'
        )

if __name__ == '__main__':
    unittest.main()

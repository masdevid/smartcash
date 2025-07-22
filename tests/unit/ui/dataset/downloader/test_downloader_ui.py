"""
Tests for downloader UI components in the SmartCash application.

This module contains unit tests for the downloader UI components,
including the main UI module and related components.
"""
import unittest
from unittest.mock import patch, MagicMock

from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule


class TestDownloaderUI(unittest.TestCase):
    """Test cases for downloader UI components."""

    def setUp(self):
        """Set up test environment."""
        self.mock_ui_components = {
            'main_container': MagicMock(),
            'progress_bar': MagicMock(),
            'status_label': MagicMock(),
            'download_button': MagicMock(),
            'cancel_button': MagicMock()
        }
        self.mock_handler = MagicMock()
        
    @patch('smartcash.ui.dataset.downloader.downloader_uimodule.DownloaderUIHandler')
    @patch('smartcash.ui.dataset.downloader.downloader_uimodule.create_downloader_ui_components')
    def test_initialize(self, mock_create_ui, mock_handler_class):
        """Test initialization of the downloader UI module."""
        # Setup mocks
        mock_create_ui.return_value = self.mock_ui_components
        mock_handler_class.return_value = self.mock_handler
        
        # Create and initialize module
        module = DownloaderUIModule()
        result = module.initialize()
        
        # Verify results
        self.assertTrue(result)
        mock_create_ui.assert_called_once()
        
    @patch('smartcash.ui.dataset.downloader.downloader_uimodule.DownloaderUIHandler')
    @patch('smartcash.ui.dataset.downloader.downloader_uimodule.create_downloader_ui_components')
    @patch('smartcash.ui.dataset.downloader.downloader_uimodule.create_download_operation')
    def test_operation_download(self, mock_create_operation, mock_create_ui, mock_handler_class):
        """Test download operation."""
        # Setup mocks
        mock_create_ui.return_value = self.mock_ui_components
        mock_handler_class.return_value = self.mock_handler
        
        # Mock the download operation
        mock_operation = MagicMock()
        mock_operation.execute.return_value = {"success": True, "file_count": 1, "total_size": "1.0MB"}
        mock_create_operation.return_value = mock_operation
        
        # Create and initialize module
        module = DownloaderUIModule()
        module.initialize()
        
        # Mock the log method to capture log messages
        module.log = MagicMock()
        
        # Call the _operation_download method directly
        result = module._operation_download()
        
        # Verify the result is None (method doesn't return anything)
        self.assertIsNone(result)
        
        # Verify the operation was created and executed
        mock_create_operation.assert_called_once()
        mock_operation.execute.assert_called_once()
        
        # Verify log messages were called
        module.log.assert_any_call("📥 Memulai download dataset...", "info")
        module.log.assert_any_call("📥 Memulai operasi download dataset", "info")
        module.log.assert_any_call("✅ Download selesai: 1 file (1.0MB)", "info")
        module.log.assert_any_call("✅ Download dataset berhasil diselesaikan", "success")


if __name__ == '__main__':
    unittest.main()

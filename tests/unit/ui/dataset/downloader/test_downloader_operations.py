"""
Tests for downloader operations in the SmartCash application.

This module contains unit tests for the downloader operations, including
file downloads and cleanup functionality.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock, Mock, ANY, PropertyMock
from pathlib import Path

# Import the operations
from smartcash.ui.dataset.downloader.operations.download_operation import DownloadOperation
from smartcash.ui.dataset.downloader.operations.download_check_operation import DownloadCheckOperation
from smartcash.ui.dataset.downloader.operations.download_cleanup_operation import DownloadCleanupOperation

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()


class TestDownloadOperations(unittest.TestCase):
    """Test cases for download operations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock UI module
        self.mock_ui = MagicMock()
        
        # Create a temporary directory for test downloads
        self.temp_dir = tempfile.mkdtemp()
        self.download_path = os.path.join(self.temp_dir, "downloads")
        self.extract_path = os.path.join(self.temp_dir, "extracted")
        
        # Create test directories
        os.makedirs(self.download_path, exist_ok=True)
        os.makedirs(self.extract_path, exist_ok=True)
        
        # Sample configuration
        self.test_config = {
            "data": {
                "roboflow": {
                    "workspace": "test-workspace",
                    "project": "test-project",
                    "version": 1,
                    "api_key": "test-api-key"
                },
                "download_path": self.download_path,
                "extract_path": self.extract_path
            }
        }
        
        # Patch the base class methods that will be called during initialization
        self.patchers = [
            patch('smartcash.ui.dataset.downloader.operations.downloader_base_operation.ColabSecretsMixin.__init__', return_value=None),
            patch('smartcash.ui.dataset.downloader.operations.downloader_base_operation.DownloaderBaseOperation._load_backend_apis', return_value={})
        ]
        
        for patcher in self.patchers:
            patcher.start()
            
    def tearDown(self):
        """Clean up after each test method."""
        # Stop all patchers
        for patcher in self.patchers:
            patcher.stop()
            
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('smartcash.ui.dataset.downloader.operations.download_operation.DownloadOperation._validate_and_prepare')
    @patch('smartcash.ui.dataset.downloader.operations.download_operation.DownloadOperation.get_backend_api')
    def test_download_operation(self, mock_get_backend_api, mock_validate_and_prepare):
        """Test file download operation."""
        # Setup mocks
        mock_download_service = MagicMock()
        mock_downloader = MagicMock()
        mock_downloader.set_progress_callback = MagicMock()
        mock_downloader.download.return_value = {
            'status': 'success',
            'file_count': 10,
            'total_size': '1.2MB',
            'download_path': '/test/path'
        }
        
        mock_get_backend_api.return_value = lambda _: mock_downloader
        
        mock_validate_and_prepare.return_value = {
            'success': True,
            'api_key': 'test-api-key',
            'backend_config': {}
        }
        
        # Create the operation
        operation = DownloadOperation(
            ui_module=self.mock_ui,
            config=self.test_config
        )
        
        # Execute the operation
        result = operation.execute()
        
        # Verify results
        self.assertTrue(result["success"])
        mock_validate_and_prepare.assert_called_once()
        mock_downloader.download.assert_called_once_with(
            workspace='test-workspace',
            project='test-project',
            version=1,
            api_key='test-api-key'
        )

    @patch('smartcash.ui.dataset.downloader.operations.download_check_operation.DownloadCheckOperation._create_dataset_scanner')
    def test_download_check_operation(self, mock_create_scanner):
        """Test download check operation."""
        # Setup test config with required fields
        test_config = {
            "download_paths": ["/test/file1.zip", "/test/file2.zip"],
            "extract_paths": ["/test/extract"],
            "data": {
                "roboflow": {
                    "workspace": "test-workspace",
                    "project": "test-project",
                    "version": 1
                }
            },
            "dataset_path": "/test/dataset",
            "dataset_name": "test-dataset"
        }
        
        # Create a mock response that matches the expected format
        mock_scan_result = {
            'status': 'success',
            'summary': {
                'total_images': 10,
                'total_labels': 10,
                'classes': 5,
                'total_files': 20,
                'total_size': '1.2MB'
            },
            'stats': {
                'images': 10,
                'labels': 10,
                'classes': 5,
                'splits': {'train': 7, 'val': 2, 'test': 1},
                'class_distribution': {}
            },
            'issues': [],
            'total_size': '1.2MB',
            'dataset_path': '/test/path',
            'exists': True,
            'file_count': 20,
            'message': 'Dataset scan completed successfully'
        }
        
        # Setup mock scanner with proper response format
        mock_scanner = MagicMock()
        
        # Mock the scanner's parallel scan method to return our mock result directly
        mock_scanner.scan_existing_dataset_parallel.return_value = mock_scan_result
        
        # Mock the set_progress_callback method
        mock_scanner.set_progress_callback = MagicMock()
        
        # Setup _create_dataset_scanner to return our mock scanner
        mock_create_scanner.return_value = mock_scanner
        
        # Create the operation
        operation = DownloadCheckOperation(
            ui_module=self.mock_ui,
            config=test_config
        )
        
        # Patch the _display_check_results and _update_summary_container methods
        with patch.object(operation, '_display_check_results') as mock_display, \
             patch.object(operation, '_update_summary_container') as mock_update, \
             patch.object(operation, 'update_progress') as mock_update_progress:
            
            # Execute the operation
            result = operation.execute()
            
            # Verify results
            self.assertTrue(result["success"])
            self.assertEqual(result["file_count"], 20)  # From the mock response
            self.assertEqual(result["total_size"], "1.2MB")
            self.assertTrue(result["exists"])
            self.assertEqual(result["dataset_path"], "/test/path")
            
            # Verify mocks were called correctly
            mock_create_scanner.assert_called_once()
            mock_scanner.scan_existing_dataset_parallel.assert_called_once()
            mock_display.assert_called_once_with(mock_scan_result)
            mock_update.assert_called_once_with(mock_scan_result)
            
            # Verify progress updates were called
            self.assertGreater(mock_update_progress.call_count, 0)

    @patch('smartcash.ui.dataset.downloader.operations.download_cleanup_operation.DownloadCleanupOperation._create_cleanup_service')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_operation(self, mock_rmtree, mock_exists, mock_create_cleanup):
        """Test cleanup operation."""
        # Setup test config
        test_config = {
            "download_paths": ["/test/file1.zip", "/test/file2.zip"],
            "extract_paths": ["/test/extract"],
            "data": {
                "roboflow": {
                    "workspace": "test-workspace",
                    "project": "test-project",
                    "version": 1
                },
                "dataset_path": "/test/dataset",
                "dataset_name": "test-dataset"
            }
        }
        
        # Setup mock cleanup service
        mock_cleanup_service = MagicMock()
        mock_cleanup_service.cleanup_dataset.return_value = {
            'status': 'success',
            'deleted_files': 2,
            'freed_space': '3 KB',
            'message': 'Cleanup completed successfully'
        }
        mock_create_cleanup.return_value = mock_cleanup_service
        
        # Setup other mocks
        mock_exists.return_value = True
        
        # Create the operation
        operation = DownloadCleanupOperation(
            ui_module=self.mock_ui,
            config=test_config
        )
        
        # Setup mock scan result with proper structure
        mock_scan_result = {
            'success': True,
            'targets': {
                'file1.zip': {
                    'path': '/test/file1.zip',
                    'size': 1024,
                    'file_count': 1,
                    'size_formatted': '1 KB'
                },
                'file2.zip': {
                    'path': '/test/file2.zip',
                    'size': 2048,
                    'file_count': 1,
                    'size_formatted': '2 KB'
                }
            },
            'summary': {
                'total_files': 2,
                'total_size': '3 KB'
            },
            'total_size': 3072,
            'message': 'Scan completed successfully'
        }
        
        # Execute the operation with the mock scan result
        with patch.object(operation, 'update_progress') as mock_update_progress:
            
            result = operation.execute(targets_result=mock_scan_result)
            
            # Verify results
            self.assertTrue(result['success'])
            self.assertEqual(result['deleted_files'], 2)
            self.assertEqual(result['freed_space'], '3 KB')
            
            # Verify mocks were called correctly
            mock_create_cleanup.assert_called_once()
            mock_cleanup_service.cleanup_dataset.assert_called_once()
            
            # Verify progress updates were called
            mock_update_progress.assert_called()


if __name__ == '__main__':
    unittest.main()

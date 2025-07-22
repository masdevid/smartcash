"""
Integration tests for the download pipeline in the SmartCash application.

This module contains integration tests that verify the complete download
pipeline, including file downloads, extraction, and cleanup.
"""
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import requests
from requests.exceptions import RequestException

from smartcash.ui.dataset.downloader.operations.download_operation import DownloadOperation
from smartcash.ui.dataset.downloader.operations.download_check_operation import DownloadCheckOperation
from smartcash.ui.dataset.downloader.operations.download_cleanup_operation import DownloadCleanupOperation


class TestDownloadPipeline(unittest.TestCase):
    """Integration tests for the download pipeline."""

    TEST_FILE_URL = "https://github.com/masdevid/smartcash/releases/download/v0.1.0/test_download.zip"
    TEST_FILE_SIZE = 1024  # Expected size in bytes

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="smartcash_test_")
        self.download_dir = os.path.join(self.test_dir, "downloads")
        self.extract_dir = os.path.join(self.test_dir, "extracted")
        
        # Create directories
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)
        
        # Test file paths
        self.download_path = os.path.join(self.download_dir, "test_download.zip")
        self.extract_path = os.path.join(self.extract_dir, "test_file.txt")
        
        # Mock UI module
        self.mock_ui = MagicMock()
        self.mock_ui.logger = MagicMock()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_complete_download_pipeline(self):
        """
        Test the complete download pipeline:
        1. Download a test file
        2. Verify the download
        3. Extract the file
        4. Verify the extracted content
        5. Clean up
        """
        # Skip test if we can't reach the test file
        try:
            response = requests.head(self.TEST_FILE_URL, timeout=5)
            if response.status_code != 200:
                self.skipTest("Test file not available")
        except RequestException:
            self.skipTest("Could not connect to test file server")

        # 1. Download the file
        download_config = {
            "url": self.TEST_FILE_URL,
            "download_path": self.download_path,
            "extract_path": self.extract_dir,
            "expected_size": self.TEST_FILE_SIZE
        }
        
        download_op = DownloadOperation(
            ui_module=self.mock_ui,
            config=download_config
        )
        
        # Mock the _update_progress method to avoid UI updates
        download_op._update_progress = MagicMock()
        
        download_result = download_op.execute()
        self.assertTrue(download_result["success"])
        self.assertTrue(os.path.exists(self.download_path))
        
        # 2. Verify the download
        check_config = {
            "download_path": self.download_path,
            "expected_size": self.TEST_FILE_SIZE
        }
        
        check_op = DownloadCheckOperation(
            ui_module=self.mock_ui,
            config=check_config
        )
        
        # Mock the _update_progress method to avoid UI updates
        check_op._update_progress = MagicMock()
        
        check_result = check_op.execute()
        self.assertTrue(check_result["success"])
        self.assertGreater(check_result.get("size", 0), 0)
        
        # 3. Extract the file (handled by DownloadOperation if it's a zip file)
        if self.download_path.endswith('.zip'):
            with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            
            # Verify extraction
            extracted_files = list(Path(self.extract_dir).rglob('*'))
            self.assertGreater(len(extracted_files), 0)
        
        # 4. Clean up
        cleanup_config = {
            "download_paths": [self.download_path],
            "extract_paths": [self.extract_dir]
        }
        
        cleanup_op = DownloadCleanupOperation(
            ui_module=self.mock_ui,
            config=cleanup_config
        )
        
        # Mock the _update_progress method to avoid UI updates
        cleanup_op._update_progress = MagicMock()
        
        cleanup_result = cleanup_op.execute()
        self.assertTrue(cleanup_result["success"])
        self.assertFalse(os.path.exists(self.download_path))
        
        # Verify cleanup
        if os.path.exists(self.extract_dir):
            self.assertEqual(len(os.listdir(self.extract_dir)), 0)


if __name__ == '__main__':
    unittest.main()

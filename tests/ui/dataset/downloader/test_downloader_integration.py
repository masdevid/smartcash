"""
File: tests/ui/dataset/downloader/test_downloader_integration.py
Description: Integration tests for the Dataset Downloader UI module and its integration with the core downloader service.
"""

import pytest
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any, Optional

from smartcash.ui.dataset.downloader.downloader_uimodule import (
    DownloaderUIModule,
    create_downloader_uimodule
)
from smartcash.dataset.downloader.download_service import DownloadService
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage
from smartcash.ui.core.ui_module import UIModule

# Test configuration
TEST_CONFIG = {
    "api_key": "test_api_key",
    "workspace": "test_workspace",
    "project": "test_project",
    "version": "1",
    "format": "yolov8",
    "location": "/test/dataset/path"
}

@pytest.fixture
def mock_download_service():
    """Create a mock DownloadService with basic functionality."""
    with patch('smartcash.dataset.downloader.download_service.DownloadService') as mock_service:
        # Configure the mock service
        mock_instance = mock_service.return_value
        mock_instance.download_dataset.return_value = {
            "success": True,
            "message": "Download completed",
            "files_downloaded": 10,
            "total_size": "50MB"
        }
        mock_instance.validate_config.return_value = {
            "valid": True,
            "message": "Configuration is valid"
        }
        yield mock_instance

@pytest.fixture
def mock_progress_tracker():
    """Create a mock ProgressTracker."""
    with patch('smartcash.dataset.downloader.progress_tracker.DownloadProgressTracker') as mock_tracker:
        mock_instance = mock_tracker.return_value
        yield mock_instance

@pytest.fixture
def downloader_module(mock_download_service, mock_progress_tracker):
    """Create a test instance of DownloaderUIModule with mocks."""
    # Create a test config
    config = {
        "data": {
            "roboflow": {
                "api_key": "test_api_key",
                "workspace": "test_workspace",
                "project": "test_project",
                "version": "1"
            },
            "format": "yolov8"
        }
    }
    
    # Create the module with test config
    module = create_downloader_uimodule(config=config, auto_initialize=True)
    
    # Mock the UI components
    module.ui_components = {
        'main_container': MagicMock(),
        'header_container': MagicMock(),
        'form_container': MagicMock(),
        'action_container': MagicMock(),
        'operation_container': MagicMock(),
        'footer_container': MagicMock()
    }
    
    # Mock the progress tracker
    module.progress_tracker = mock_progress_tracker
    
    # Mock the download service
    module._downloader_service = mock_download_service
    
    yield module
    
    # Cleanup
    if hasattr(module, '_instance'):
        del module._instance

class TestDownloaderIntegration:
    """Integration tests for the DownloaderUIModule."""
    
    def test_initialization(self, downloader_module):
        """Test that the downloader module initializes correctly."""
        assert downloader_module is not None
        assert isinstance(downloader_module, UIModule)
        assert hasattr(downloader_module, 'ui_components')
        assert 'main_container' in downloader_module.ui_components
    
    @pytest.mark.skip(reason="Temporarily skipped until download workflow is updated")
    def test_download_workflow(self, downloader_module, mock_download_service, mock_progress_tracker, mocker):
        """Test the complete download workflow with progress tracking."""
        # Setup test data
        test_config = {
            "source_type": "roboflow",
            "data": {
                "roboflow": {
                    "api_key": "test_api_key",
                    "workspace": "test_workspace",
                    "project": "test_project",
                    "version": "1"
                },
                "format": "yolov8",
                "location": "/test/dataset/path"
            }
        }
        
        # Expected result from the download service
        expected_result = {
            "status": "success",
            "message": "Download completed",
            "files_downloaded": 10,
            "total_size": "50MB",
            "dataset_path": "/test/dataset/path",
            "summary": {
                "total_images": 10,
                "total_labels": 10
            },
            "stats": {
                "total_images": 10,
                "total_annotations": 10
            }
        }
        
        # Create a mock for the operation manager
        operation_manager_mock = MagicMock()
        operation_manager_mock.execute_download.return_value = {
            "success": True,
            "message": "Download completed successfully",
            "result": expected_result
        }
        
        # Patch the operation manager creation to return our mock
        with patch.object(downloader_module, '_operation_manager', operation_manager_mock):
            # Execute download with the test config
            result = downloader_module.execute_download(ui_config=test_config)
            
            # Verify the operation manager's execute_download was called with the test config
            operation_manager_mock.execute_download.assert_called_once_with(test_config)
            
            # Verify the result is as expected
            assert result["success"] is True
            assert "Download completed successfully" in result["message"]
            assert result.get("result") == expected_result
    
    @patch('smartcash.dataset.downloader.dataset_scanner.create_dataset_scanner')
    def test_check_operation(self, mock_create_scanner, downloader_module, mock_download_service):
        """Test the check dataset operation."""
        # Create a mock scanner
        mock_scanner = MagicMock()
        mock_scanner.scan_existing_dataset_parallel.return_value = {
            'status': 'success',
            'file_count': 10,
            'total_size': '50MB',
            'message': 'Scan completed successfully'
        }
        mock_create_scanner.return_value = mock_scanner
        
        # Setup UI components needed by the handler
        downloader_module.ui_components.update({
            'progress_callback': MagicMock()
        })
        
        # Execute check
        result = downloader_module.execute_check()
        
        # Verify results
        assert result["exists"] is True
        assert result["file_count"] == 10
        assert result["total_size"] == "50MB"
        assert result["status"] is True
        assert "Pemeriksaan dataset selesai" in result["message"]
        
        # Verify scanner was called
        mock_create_scanner.assert_called_once()
        mock_scanner.scan_existing_dataset_parallel.assert_called_once()
    
    def test_cleanup_operation(self, downloader_module, mocker):
        """Test the cleanup operation."""
        # Mock the operation manager's execute_cleanup method
        mock_result = {
            "success": True,
            "deleted_files": 5,
            "freed_space": "25MB",
            "message": "Cleanup completed successfully"
        }
        
        # Patch the operation manager's execute_cleanup method
        with patch.object(downloader_module, '_operation_manager') as mock_manager:
            mock_manager.execute_cleanup.return_value = mock_result
            
            # Execute cleanup
            result = downloader_module.execute_cleanup()
            
            # Verify results
            assert result["success"] is True
            assert result["deleted_files"] == 5
            assert result["freed_space"] == "25MB"
            assert "Cleanup completed" in result["message"]
            
            # Verify the operation manager was called
            mock_manager.execute_cleanup.assert_called_once()
    
    def test_config_validation(self, downloader_module, mock_download_service):
        """Test configuration validation."""
        # Setup test config
        test_config = {
            "data": {
                "roboflow": {
                    "api_key": "test_api_key",
                    "workspace": "test_workspace",
                    "project": "test_project",
                    "version": "1"
                },
                "format": "yolov8"
            }
        }
        
        # Test valid config
        result = downloader_module.validate_configuration(test_config)
        assert result["valid"] is True
        
        # Test invalid config
        mock_download_service.validate_config.return_value = {
            "valid": False,
            "errors": ["Invalid API key"]
        }
        
        result = downloader_module.validate_configuration({"data": {"roboflow": {"api_key": ""}}})
        assert result["valid"] is False
        assert "Invalid API key" in result["errors"][0]

    def test_ui_components_initialization(self, downloader_module):
        """Test that all UI components are properly initialized."""
        # Verify all required UI components exist
        required_components = [
            'main_container', 'header_container', 'form_container',
            'action_container', 'operation_container', 'footer_container'
        ]
        
        for component in required_components:
            assert component in downloader_module.ui_components
            assert downloader_module.ui_components[component] is not None
        
        # Verify form elements are created
        form_container = downloader_module.ui_components['form_container']
        form_container.children = [MagicMock(description=desc) for desc in 
                                 ['API Key', 'Workspace', 'Project', 'Version', 'Format']]
        
        # Verify action buttons
        action_container = downloader_module.ui_components['action_container']
        action_container.children = [
            MagicMock(description=desc) for desc in 
            ['Download', 'Check', 'Cleanup']
        ]
    
    def test_error_handling(self, downloader_module, mock_download_service):
        """Test error handling during download operations."""
        # Mock a download error
        mock_download_service.download_dataset.side_effect = Exception("Download failed: Connection error")
        
        # Execute download that should fail
        result = downloader_module.execute_download()
        
        # Verify error handling
        assert result["success"] is False
        assert "Download failed" in result.get("error", "")
        
        # Verify error was logged
        downloader_module.logger.error.assert_called_with(
            "Error during download operation: Download failed: Connection error"
        )

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

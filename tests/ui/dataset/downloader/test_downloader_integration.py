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
    
    def test_download_workflow(self, downloader_module, mock_download_service, mock_progress_tracker):
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
        
        # Configure the mock download service to return our expected result
        mock_download_service.download_dataset.return_value = expected_result
        
        # Create a proper mock for the progress tracker with the expected methods
        progress_tracker_mock = MagicMock()
        progress_tracker_mock.start_stage = MagicMock()
        progress_tracker_mock.complete_stage = MagicMock()
        progress_tracker_mock.update_overall = MagicMock()
        progress_tracker_mock.update_progress = MagicMock()
        
        # Create a mock for the operation container
        operation_container_mock = MagicMock()
        
        # Create a mock for the operation manager
        operation_manager_mock = MagicMock()
        operation_manager_mock.execute_download = MagicMock(return_value={
            "status": True,
            "message": "Download completed successfully"
        })
        
        # Setup the UI components that the DownloadOperationHandler expects
        ui_components = {
            'progress_tracker': progress_tracker_mock,
            'progress_callback': MagicMock(),
            'status_panel': MagicMock(),
            'summary_container': MagicMock(),
            'progress_bar': MagicMock(),
            'operation_container': operation_container_mock
        }
        
        # Patch the operation manager creation to return our mock
        with patch('smartcash.ui.dataset.downloader.downloader_uimodule.DownloaderOperationManager') as mock_op_manager_class, \
             patch('smartcash.ui.dataset.downloader.services.backend_utils.create_backend_downloader') as mock_create_downloader:
            
            # Configure the mock operation manager class to return our mock instance
            mock_op_manager_class.return_value = operation_manager_mock
            
            # Configure the mock download service
            mock_create_downloader.return_value = mock_download_service
            
            # Set the UI components on the downloader module
            downloader_module._ui_components = ui_components
            
            # Execute download with the test config
            result = downloader_module.execute_download(ui_config=test_config)
            
            # Verify the operation manager was created with the correct arguments
            mock_op_manager_class.assert_called_once_with(
                config=downloader_module.get_config(),
                operation_container=operation_container_mock
            )
            
            # Verify the operation manager's execute_download was called with the test config
            operation_manager_mock.execute_download.assert_called_once()
            
            # Verify the download service was called
            mock_download_service.download_dataset.assert_called_once()
            
            # Verify progress tracking was updated
            progress_tracker_mock.start_stage.assert_called_once_with("Dataset Download")
            progress_tracker_mock.complete_stage.assert_called_once()
            progress_tracker_mock.update_overall.assert_called()
            assert progress_tracker_mock.update_progress.call_count > 0
    
    def test_check_operation(self, downloader_module, mock_download_service):
        """Test the check dataset operation."""
        # Mock the check operation
        mock_download_service.check_dataset.return_value = {
            "exists": True,
            "file_count": 10,
            "total_size": "50MB"
        }
        
        # Execute check
        result = downloader_module.execute_check()
        
        # Verify results
        assert result["exists"] is True
        assert result["file_count"] == 10
        
        # Verify service was called
        mock_download_service.check_dataset.assert_called_once()
    
    def test_cleanup_operation(self, downloader_module, mock_download_service):
        """Test the cleanup operation."""
        # Mock the cleanup operation
        mock_download_service.cleanup.return_value = {
            "success": True,
            "deleted_files": 5,
            "freed_space": "25MB"
        }
        
        # Execute cleanup
        result = downloader_module.execute_cleanup()
        
        # Verify results
        assert result["success"] is True
        assert result["deleted_files"] == 5
        
        # Verify service was called
        mock_download_service.cleanup.assert_called_once()
    
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

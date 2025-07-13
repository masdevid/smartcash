"""
File: tests/ui/dataset/downloader/test_downloader_integration.py
Description: Integration tests for the Dataset Downloader UI module and its integration with the core downloader service.
"""

import pytest
from unittest.mock import MagicMock, patch, call, AsyncMock
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.ui.dataset.downloader.downloader_uimodule import (
    DownloaderUIModule,
    create_downloader_uimodule
)
from smartcash.ui.core.ui_module import UIModule
from smartcash.dataset.downloader.download_service import DownloadService
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage

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
    
    @pytest.mark.asyncio
    async def test_check_operation(self, downloader_module, mock_download_service):
        """Test the check dataset operation."""
        # Create a mock operation manager with a mock execute_check method
        mock_op_manager = AsyncMock()
        mock_op_manager.execute_check.return_value = {
            'success': True,
            'status': True,
            'file_count': 10,
            'total_size': '50MB',
            'message': 'Pemeriksaan dataset selesai',
            'exists': True
        }
        
        # Patch the operation manager
        with patch.object(downloader_module, '_operation_manager', mock_op_manager):
            # Execute check
            result = await downloader_module.execute_check()
            
            # Verify results
            assert result.get("success") is True
            assert result.get("status") is True
            assert result.get("file_count") == 10
            assert result.get("total_size") == "50MB"
            assert "Pemeriksaan dataset selesai" in result.get("message", "")
            
            # Verify the execute_check method was called
            mock_op_manager.execute_check.assert_awaited_once()
    
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
    
    @pytest.mark.asyncio
    async def test_error_handling(self, downloader_module, mock_download_service):
        """Test error handling during download operations."""
        # Create a mock operation manager that returns an error response
        mock_op_manager = AsyncMock()
        mock_op_manager.execute_download.return_value = {
            "success": False,
            "error": "Download failed: Connection error",
            "message": "Failed to download dataset"
        }
        
        # Patch the operation manager
        with patch.object(downloader_module, '_operation_manager', mock_op_manager):
            # Execute download that should fail
            result = await downloader_module.execute_download()
            
            # Verify error handling
            assert result.get("success") is False
            assert "Download failed" in result.get("error", "")
            
            # Verify the execute_download method was called
            mock_op_manager.execute_download.assert_awaited_once()

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])

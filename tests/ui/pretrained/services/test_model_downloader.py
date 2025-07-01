"""
File: tests/ui/pretrained/services/test_model_downloader.py
Deskripsi: Tests for model_downloader.py
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call, ANY
from typing import Optional, Callable

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module we're testing
from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
from smartcash.ui.pretrained.utils.progress_adapter import PretrainedProgressAdapter
from smartcash.ui.types import ProgressTrackerType, StatusCallback

# Create a mock class for PretrainedProgressAdapter that will be used in the test
class MockPretrainedProgressAdapter:
    def __init__(self, *args, **kwargs):
        self.get_status_callback = MagicMock(return_value=MagicMock())
        self.update_progress = MagicMock()
        self.set_status = MagicMock()
        self.complete = MagicMock()
        self.fail = MagicMock()

# Patch the PretrainedProgressAdapter at the module level where it's imported in model_downloader
@pytest.fixture(autouse=True)
def mock_imports():
    with patch('smartcash.ui.pretrained.services.model_downloader.requests') as mock_requests, \
         patch('smartcash.ui.pretrained.services.model_downloader.logging') as mock_logging, \
         patch('smartcash.ui.pretrained.services.model_downloader.os') as mock_os, \
         patch('smartcash.ui.pretrained.services.model_downloader.hashlib') as mock_hashlib, \
         patch('smartcash.ui.pretrained.services.model_downloader.ThreadPoolExecutor') as mock_executor, \
         patch('smartcash.ui.pretrained.services.model_downloader.PretrainedProgressAdapter', new=MockPretrainedProgressAdapter) as mock_progress_adapter:
        
        # Configure the mocks
        mock_os.path = MagicMock()
        mock_os.path.exists.return_value = False
        mock_os.path.join.return_value = "/tmp/test_model.pt"
        mock_os.path.getsize.return_value = 1024  # 1KB file size
        
        # Mock the response for requests.get
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_response.headers = {'content-length': '100'}
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        
        # Mock hashlib.md5
        mock_md5 = MagicMock()
        mock_md5.hexdigest.return_value = 'test_md5_hash'
        mock_hashlib.md5.return_value = mock_md5
        
        # Create a mock instance of the progress adapter
        mock_adapter_instance = MockPretrainedProgressAdapter()
        mock_progress_adapter.return_value = mock_adapter_instance
        
        yield {
            'mock_requests': mock_requests,
            'mock_logging': mock_logging,
            'mock_os': mock_os,
            'mock_hashlib': mock_hashlib,
            'mock_md5': mock_md5,
            'mock_executor': mock_executor,
            'mock_progress_adapter': mock_progress_adapter,
            'mock_adapter_instance': mock_adapter_instance,
            'mock_response': mock_response
        }

def test_model_downloader_initialization():
    """Test that PretrainedModelDownloader can be instantiated."""
    # Create an instance of the downloader
    downloader = PretrainedModelDownloader()
    
    # Verify the instance was created
    assert downloader is not None
    assert isinstance(downloader, PretrainedModelDownloader)
    
    # Verify default attributes
    assert hasattr(downloader, '_logger_bridge')
    assert hasattr(downloader, '_progress_tracker')
    assert hasattr(downloader, '_download_urls')
    assert 'yolov5s' in downloader._download_urls

def test_set_progress_callbacks_with_progress_tracker(mock_imports):
    """Test setting progress callbacks with a progress tracker."""
    # Get the mock adapter instance from the fixture
    mock_adapter_instance = mock_imports['mock_adapter_instance']
    
    # Test case 1: Initialize with progress_tracker in constructor
    downloader = PretrainedModelDownloader(progress_tracker=mock_adapter_instance)
    
    # Verify the progress tracker was set in constructor
    assert downloader._progress_tracker is not None
    
    # Test case 2: Set callbacks using set_progress_callbacks
    progress_cb = MagicMock()
    status_cb = MagicMock()
    
    # Set the callbacks
    downloader.set_progress_callbacks(progress_cb, status_cb)
    
    # Verify the progress tracker is still set
    assert downloader._progress_tracker is not None
    
    # Verify the callbacks were set directly
    assert downloader._progress_callback == progress_cb
    assert downloader._status_callback == status_cb
    
    # Test case 3: Set progress tracker via set_progress_callbacks with a progress tracker
    # Configure the mock to return a status callback
    mock_adapter_instance.get_status_callback.return_value = "mock_status_from_adapter"
    
    # Set the progress tracker via set_progress_callbacks
    downloader.set_progress_callbacks(mock_adapter_instance, None)
    
    # Verify the progress tracker was set
    assert downloader._progress_tracker is not None
    
    # Verify get_status_callback was called on the adapter
    mock_adapter_instance.get_status_callback.assert_called_once()
    
    # Verify the status callback was set from the adapter
    assert downloader._status_callback == "mock_status_from_adapter"
    
    # Test case 4: Set callbacks directly without a progress tracker
    downloader = PretrainedModelDownloader()
    downloader.set_progress_callbacks(progress_cb, status_cb)
    
    # Verify callbacks were set directly and no progress tracker is set
    assert downloader._progress_callback == progress_cb
    assert downloader._status_callback == status_cb
    assert downloader._progress_tracker is None

def test_set_logger_bridge():
    """Test setting the logger bridge."""
    # Create a mock logger bridge
    mock_logger_bridge = MagicMock()
    
    # Create an instance of the downloader
    downloader = PretrainedModelDownloader()
    
    # Set the logger bridge
    downloader.set_logger_bridge(mock_logger_bridge)
    
    # Verify the logger bridge was set
    assert downloader._logger_bridge == mock_logger_bridge

def test_safe_callback():
    """Test the _safe_callback method with a callback that raises an exception."""
    # Create an instance of the downloader
    downloader = PretrainedModelDownloader()
    
    # Create a mock callback that raises an exception
    mock_callback = MagicMock(side_effect=Exception("Test exception"))
    
    # Call the _safe_callback method
    downloader._safe_callback(mock_callback, "test_arg", kwarg="test_kwarg")
    
    # Verify the callback was called with the correct arguments
    mock_callback.assert_called_once_with("test_arg", kwarg="test_kwarg")
    
    # Verify the error was logged (we can check the mock logger if needed)
    assert True  # Just verify we got here without exceptions

"""
File: tests/ui/dataset/downloader/conftest.py
Description: Pytest configuration and fixtures for downloader integration tests.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to Python path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Create test directories
    test_dirs = [
        "datasets",
        "downloads",
        "logs"
    ]
    
    for dir_name in test_dirs:
        os.makedirs(TEST_DATA_DIR / dir_name, exist_ok=True)
    
    yield  # Test runs happen here
    
    # Cleanup test directories
    for dir_name in test_dirs:
        dir_path = TEST_DATA_DIR / dir_name
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)

@pytest.fixture
def mock_ui_components():
    """Create mock UI components for testing."""
    return {
        'main_container': MagicMock(),
        'header_container': MagicMock(),
        'form_container': MagicMock(),
        'action_container': MagicMock(),
        'operation_container': MagicMock(),
        'footer_container': MagicMock(),
        'progress_bar': MagicMock(),
        'status_label': MagicMock(),
        'log_output': MagicMock()
    }

@pytest.fixture
def mock_download_service():
    """Create a mock DownloadService."""
    with patch('smartcash.dataset.downloader.download_service.DownloadService') as mock_service:
        instance = mock_service.return_value
        instance.download_dataset.return_value = {
            "success": True,
            "message": "Download completed",
            "files_downloaded": 10,
            "total_size": "50MB"
        }
        instance.check_dataset.return_value = {
            "exists": True,
            "file_count": 10,
            "total_size": "50MB"
        }
        instance.cleanup.return_value = {
            "success": True,
            "deleted_files": 5,
            "freed_space": "25MB"
        }
        instance.validate_config.return_value = {
            "valid": True,
            "message": "Configuration is valid"
        }
        yield instance

@pytest.fixture
def mock_progress_tracker():
    """Create a mock ProgressTracker."""
    with patch('smartcash.dataset.downloader.progress_tracker.DownloadProgressTracker') as mock_tracker:
        instance = mock_tracker.return_value
        yield instance

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    with patch('smartcash.ui.logger.get_module_logger') as mock_logger:
        instance = MagicMock()
        mock_logger.return_value = instance
        yield instance

@pytest.fixture
def downloader_config():
    """Return a sample downloader configuration."""
    return {
        "data": {
            "roboflow": {
                "api_key": "test_api_key",
                "workspace": "test_workspace",
                "project": "test_project",
                "version": "1"
            },
            "format": "yolov8",
            "location": str(TEST_DATA_DIR / "datasets")
        },
        "ui": {
            "show_progress": True,
            "show_logs": True
        }
    }

@pytest.fixture
def downloader_module(mock_download_service, mock_progress_tracker, mock_ui_components, downloader_config):
    """Create a test instance of DownloaderUIModule with mocks."""
    from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule
    
    # Create module with test config
    module = DownloaderUIModule(config=downloader_config)
    
    # Setup mocks
    module.ui_components = mock_ui_components
    module._downloader_service = mock_download_service
    module.progress_tracker = mock_progress_tracker
    
    # Mock the config handler
    module._config_handler = MagicMock()
    module._config_handler.get_config.return_value = downloader_config
    
    # Mock the operation manager
    module._operation_manager = MagicMock()
    
    yield module
    
    # Cleanup
    if hasattr(module, '_instance'):
        del module._instance

# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv('ROBOFLOW_API_KEY', 'test_api_key')
    monkeypatch.setenv('DATASET_DOWNLOAD_DIR', str(TEST_DATA_DIR / 'datasets'))
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')

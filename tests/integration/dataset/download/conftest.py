"""
File: tests/integration/dataset/download/conftest.py
Deskripsi: Fixtures untuk integration test modul download dataset
"""
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

# Import komponen yang diperlukan untuk mocking
from smartcash.ui.dataset.download.handlers.download_config_setup import setup_download_config_handlers
from smartcash.ui.dataset.download.handlers.download_handlers_setup import setup_download_handlers
from smartcash.ui.dataset.download.handlers.download_progress_setup import setup_download_progress_handlers
from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService

@pytest.fixture(scope="module")
def temp_download_dir():
    """Fixture untuk direktori download sementara."""
    temp_dir = tempfile.mkdtemp(prefix="smartcash_test_download_")
    yield temp_dir
    # Cleanup setelah test selesai
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_ui_components(temp_download_dir):
    """Fixture untuk komponen UI yang dimock."""
    return {
        'logger': MagicMock(),
        'config': {},
        'progress_container': MagicMock(),
        'progress_bar': MagicMock(),
        'status_output': MagicMock(),
        'download_button': MagicMock(),
        'reset_button': MagicMock(),
        'check_button': MagicMock(),
        'dataset_dir': temp_download_dir,
        '_progress_setup_complete': False,
        '_progress_setup_error': None
    }

@pytest.fixture
def mock_roboflow_downloader():
    """Fixture untuk mock RoboflowDownloader."""
    with patch('smartcash.dataset.services.downloader.roboflow_downloader.RoboflowDownloader') as mock:
        yield mock

@pytest.fixture
def mock_download_service():
    """Fixture untuk mock DownloadService."""
    with patch('smartcash.dataset.services.downloader.download_service.DownloadService') as mock:
        yield mock

@pytest.fixture
def setup_download_components(mock_ui_components):
    """Setup semua komponen download untuk testing."""
    # Setup config handlers
    setup_download_config_handlers(mock_ui_components)
    
    # Setup progress handlers
    setup_download_progress_handlers(mock_ui_components)
    
    # Setup action handlers
    setup_download_handlers(mock_ui_components)
    
    return mock_ui_components

@pytest.fixture
def ui_download_service(mock_ui_components):
    """Fixture untuk UIDownloadService dengan mock dependencies."""
    service = UIDownloadService(mock_ui_components)
    return service

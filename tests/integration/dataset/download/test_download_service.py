"""
File: tests/integration/dataset/download/test_download_service.py
Deskripsi: Integration test untuk download service
"""
import pytest
from unittest.mock import MagicMock, patch, call
import os
import shutil
import tempfile
from pathlib import Path

from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader

class TestDownloadServiceIntegration:
    """Test class untuk integrasi download service."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup untuk test."""
        self.temp_dir = tempfile.mkdtemp(prefix="smartcash_test_download_")
        self.api_key = "test_api_key"
        self.workspace = "test_workspace"
        self.project = "test_project"
        self.version = 1
        self.format = "yolov5"
        
        # Setup mock untuk RoboflowDownloader
        self.mock_roboflow = MagicMock(spec=RoboflowDownloader)
        self.mock_roboflow.validate_credentials.return_value = True
        self.mock_roboflow.download.return_value = True
        
        # Buat instance DownloadService dengan mock
        with patch('smartcash.dataset.services.downloader.download_service.RoboflowDownloader', 
                  return_value=self.mock_roboflow):
            self.service = DownloadService(
                api_key=self.api_key,
                workspace=self.workspace,
                project=self.project,
                version=self.version,
                format=self.format,
                location=self.temp_dir
            )
        
        yield
        
        # Cleanup
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_download_success(self):
        """Test download berhasil."""
        # Panggil method download
        result = self.service.download()
        
        # Verifikasi hasil
        assert result is True
        self.mock_roboflow.download.assert_called_once()
    
    def test_download_with_callback(self):
        """Test download dengan callback progress."""
        # Setup mock callback
        mock_callback = MagicMock()
        
        # Panggil method download dengan callback
        result = self.service.download(progress_callback=mock_callback)
        
        # Verifikasi hasil
        assert result is True
        self.mock_roboflow.download.assert_called_once()
        
        # Verifikasi callback dipanggil dengan progress
        assert mock_callback.call_count > 0
        
        # Verifikasi argumen callback
        call_args = mock_callback.call_args[0][0]
        assert 'progress' in call_args
        assert 'status' in call_args
    
    def test_validate_credentials_success(self):
        """Test validasi kredensial berhasil."""
        # Panggil method validate_credentials
        result = self.service.validate_credentials()
        
        # Verifikasi hasil
        assert result is True
        self.mock_roboflow.validate_credentials.assert_called_once_with(
            self.api_key, self.workspace, self.project, self.version
        )
    
    def test_validate_credentials_failure(self):
        """Test validasi kredensial gagal."""
        # Setup mock untuk mengembalikan False
        self.mock_roboflow.validate_credentials.return_value = False
        
        # Panggil method validate_credentials
        result = self.service.validate_credentials()
        
        # Verifikasi hasil
        assert result is False
    
    def test_download_with_invalid_credentials(self):
        """Test download dengan kredensial tidak valid."""
        # Setup mock untuk mengembalikan False saat validasi
        self.mock_roboflow.validate_credentials.return_value = False
        
        # Panggil method download
        result = self.service.download()
        
        # Verifikasi hasil
        assert result is False
        self.mock_roboflow.download.assert_not_called()
    
    def test_download_with_error(self):
        """Test download dengan error."""
        # Setup mock untuk melempar exception
        self.mock_roboflow.download.side_effect = Exception("Download error")
        
        # Panggil method download dan verifikasi exception
        with pytest.raises(Exception):
            self.service.download()
        
        # Verifikasi download dipanggil
        self.mock_roboflow.download.assert_called_once()
    
    def test_download_with_custom_location(self):
        """Test download dengan lokasi kustom."""
        custom_dir = os.path.join(self.temp_dir, "custom_location")
        
        # Buat service baru dengan lokasi kustom
        with patch('smartcash.dataset.services.downloader.download_service.RoboflowDownloader', 
                  return_value=self.mock_roboflow):
            service = DownloadService(
                api_key=self.api_key,
                workspace=self.workspace,
                project=self.project,
                version=self.version,
                format=self.format,
                location=custom_dir
            )
        
        # Panggil method download
        result = service.download()
        
        # Verifikasi hasil
        assert result is True
        self.mock_roboflow.download.assert_called_once()
        
        # Verifikasi direktori dibuat
        assert os.path.exists(custom_dir)

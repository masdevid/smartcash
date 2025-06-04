"""
File: tests/integration/dataset/download/test_ui_download_service.py
Deskripsi: Integration test untuk UI download service
"""
import pytest
from unittest.mock import MagicMock, patch, call
import os
import json
from pathlib import Path

from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService

class TestUIDownloadServiceIntegration:
    """Test class untuk integrasi UI download service."""
    
    @pytest.fixture
    def mock_ui_components(self):
        """Fixture untuk mock UI components."""
        return {
            'config': {},
            'progress_bar': MagicMock(),
            'status_output': MagicMock(),
            'download_button': MagicMock(),
            'check_button': MagicMock(),
            'reset_button': MagicMock(),
            'logger': MagicMock()
        }
    
    @pytest.fixture
    def ui_service(self, mock_ui_components):
        """Fixture untuk UIDownloadService dengan mock dependencies."""
        with patch('smartcash.dataset.services.downloader.download_service.DownloadService') as mock_service:
            mock_service.return_value.download.return_value = True
            mock_service.return_value.validate_credentials.return_value = True
            
            service = UIDownloadService(mock_ui_components)
            service._download_service = mock_service.return_value
            
            yield service, mock_ui_components, mock_service.return_value
    
    def test_initialize_service(self, mock_ui_components):
        """Test inisialisasi UIDownloadService."""
        with patch('smartcash.dataset.services.downloader.download_service.DownloadService') as mock_service:
            service = UIDownloadService(mock_ui_components)
            
            assert service._ui == mock_ui_components
            assert service._download_service is not None
    
    def test_validate_credentials_success(self, ui_service):
        """Test validasi kredensial berhasil."""
        service, ui_components, mock_service = ui_service
        
        # Setup config
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1,
            'format': 'yolov5'
        })
        
        # Panggil method
        result = service.validate_credentials()
        
        # Verifikasi hasil
        assert result is True
        mock_service.validate_credentials.assert_called_once()
        
        # Verifikasi UI diupdate
        ui_components['status_output'].append_stdout.assert_called()
    
    def test_validate_credentials_missing_fields(self, ui_service):
        """Test validasi kredensial dengan field yang hilang."""
        service, ui_components, _ = ui_service
        
        # Setup config tanpa field yang diperlukan
        ui_components['config'].update({})
        
        # Panggil method
        result = service.validate_credentials()
        
        # Verifikasi hasil
        assert result is False
        ui_components['status_output'].append_stderr.assert_called()
    
    def test_download_success(self, ui_service):
        """Test download berhasil."""
        service, ui_components, mock_service = ui_service
        
        # Setup config
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1,
            'format': 'yolov5',
            'location': '/test/path'
        })
        
        # Panggil method
        result = service.download()
        
        # Verifikasi hasil
        assert result is True
        mock_service.download.assert_called_once()
        
        # Verifikasi UI diupdate
        ui_components['progress_bar'].value = 0
        ui_components['status_output'].append_stdout.assert_called()
    
    def test_download_with_progress(self, ui_service):
        """Test download dengan progress update."""
        service, ui_components, mock_service = ui_service
        
        # Setup mock untuk progress callback
        progress_data = {
            'progress': 50,
            'status': 'Downloading',
            'message': 'Download in progress'
        }
        
        def mock_download(progress_callback=None):
            if progress_callback:
                progress_callback(progress_data)
            return True
            
        mock_service.download.side_effect = mock_download
        
        # Setup config
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1,
            'format': 'yolov5',
            'location': '/test/path'
        })
        
        # Panggil method
        result = service.download()
        
        # Verifikasi hasil
        assert result is True
        
        # Verifikasi progress diupdate
        ui_components['progress_bar'].value = progress_data['progress']
        ui_components['status_output'].append_stdout.assert_called()
    
    def test_download_error(self, ui_service):
        """Test download dengan error."""
        service, ui_components, mock_service = ui_service
        
        # Setup mock untuk melempar exception
        mock_service.download.side_effect = Exception("Download error")
        
        # Setup config
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1,
            'format': 'yolov5',
            'location': '/test/path'
        })
        
        # Panggil method dan verifikasi exception
        with pytest.raises(Exception):
            service.download()
        
        # Verifikasi error handling
        ui_components['status_output'].append_stderr.assert_called()
    
    def test_reset_ui(self, ui_service):
        """Test reset UI."""
        service, ui_components, _ = ui_service
        
        # Set beberapa nilai awal
        ui_components['progress_bar'].value = 50
        ui_components['config'] = {'key': 'value'}
        
        # Panggil method
        service.reset_ui()
        
        # Verifikasi UI direset
        assert ui_components['progress_bar'].value == 0
        assert ui_components['config'] == {}
        ui_components['status_output'].clear_output.assert_called()
        ui_components['status_output'].append_stdout.assert_called_with("UI berhasil direset")

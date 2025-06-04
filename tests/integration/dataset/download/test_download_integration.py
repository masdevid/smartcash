"""
File: tests/integration/dataset/download/test_download_integration.py
Deskripsi: Integration test untuk modul download dataset
"""
import pytest
from unittest.mock import MagicMock, call, ANY
import json
import os
from pathlib import Path

class TestDownloadIntegration:
    """Test class untuk menguji integrasi modul download dataset."""
    
    def test_download_config_setup(self, setup_download_components):
        """Test setup konfigurasi download."""
        ui_components = setup_download_components
        
        # Verifikasi komponen config terdaftar dengan benar
        assert 'config' in ui_components
        assert 'dataset_dir' in ui_components
        assert 'api_key' in ui_components['config']
        
        # Verifikasi handler terdaftar
        assert 'on_download_click' in ui_components
        assert 'on_reset_click' in ui_components
        assert 'on_check_click' in ui_components
    
    def test_download_progress_setup(self, setup_download_components):
        """Test setup progress tracking."""
        ui_components = setup_download_components
        
        # Verifikasi progress tracking terdaftar
        assert '_progress_setup_complete' in ui_components
        assert ui_components['_progress_setup_complete'] is True
        
        # Verifikasi progress components
        assert 'progress_container' in ui_components
        assert 'progress_bar' in ui_components
        assert 'status_output' in ui_components
    
    @pytest.mark.parametrize("api_key,workspace,project,version,expected_valid", [
        ("valid_key", "test_ws", "test_proj", 1, True),
        ("", "test_ws", "test_proj", 1, False),  # Empty API key
        ("valid_key", "", "test_proj", 1, False),  # Empty workspace
        ("valid_key", "test_ws", "", 1, False),    # Empty project
        ("valid_key", "test_ws", "test_proj", 0, False),  # Invalid version
    ])
    def test_download_validation(
        self, setup_download_components, mock_roboflow_downloader, 
        api_key, workspace, project, version, expected_valid
    ):
        """Test validasi parameter download."""
        ui_components = setup_download_components
        
        # Setup mock return value
        mock_roboflow_downloader.return_value.validate_credentials.return_value = expected_valid
        
        # Set parameter
        ui_components['config'].update({
            'api_key': api_key,
            'workspace': workspace,
            'project': project,
            'version': version
        })
        
        # Panggil handler validasi
        validation_result = ui_components['on_check_click'](None)
        
        # Verifikasi hasil validasi
        if expected_valid:
            assert validation_result is True
            mock_roboflow_downloader.return_value.validate_credentials.assert_called_once_with(api_key, workspace, project, version)
        else:
            assert validation_result is False
    
    def test_download_process(self, setup_download_components, mock_roboflow_downloader):
        """Test proses download dataset."""
        ui_components = setup_download_components
        
        # Setup mock
        mock_downloader = MagicMock()
        mock_roboflow_downloader.return_value = mock_downloader
        
        # Set config yang valid
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1,
            'format': 'yolov5',
            'location': ui_components['dataset_dir']
        })
        
        # Panggil handler download
        ui_components['on_download_click'](None)
        
        # Verifikasi downloader dipanggil dengan parameter yang benar
        mock_roboflow_downloader.assert_called_once_with(
            api_key='test_key',
            workspace='test_ws',
            project='test_proj',
            version=1,
            format='yolov5',
            location=ui_components['dataset_dir']
        )
        
        # Verifikasi progress tracking dipanggil
        assert ui_components['progress_bar'].value == 0
        assert 'Download started' in str(ui_components['status_output'].append_stdout.call_args)
    
    def test_download_error_handling(self, setup_download_components, mock_roboflow_downloader):
        """Test penanganan error saat download gagal."""
        ui_components = setup_download_components
        
        # Setup mock untuk melempar exception
        mock_roboflow_downloader.side_effect = Exception("Download failed")
        
        # Set config yang valid
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1,
            'format': 'yolov5',
            'location': ui_components['dataset_dir']
        })
        
        # Panggil handler download dan tangkap exception
        with pytest.raises(Exception):
            ui_components['on_download_click'](None)
        
        # Verifikasi error handling
        assert 'Error' in str(ui_components['status_output'].append_stderr.call_args)
    
    def test_reset_functionality(self, setup_download_components):
        """Test fungsi reset konfigurasi."""
        ui_components = setup_download_components
        
        # Set beberapa nilai
        ui_components['config'].update({
            'api_key': 'test_key',
            'workspace': 'test_ws',
            'project': 'test_proj',
            'version': 1
        })
        ui_components['progress_bar'].value = 50
        
        # Panggil handler reset
        ui_components['on_reset_click'](None)
        
        # Verifikasi reset
        assert ui_components['config'] == {}
        assert ui_components['progress_bar'].value == 0
        assert 'Reset' in str(ui_components['status_output'].append_stdout.call_args)

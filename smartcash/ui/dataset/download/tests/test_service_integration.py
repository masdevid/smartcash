"""
File: smartcash/ui/dataset/download/tests/test_service_integration.py
Deskripsi: Test suite untuk integrasi antara UI dan service download dataset
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
from typing import Dict, Any
import inspect
from ipywidgets import Layout

# Tambahkan path ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from smartcash.ui.dataset.download.handlers.download_handler import (
    _download_from_roboflow,
    execute_download
)
from smartcash.dataset.manager import DatasetManager
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.ui.dataset.download.utils.notification_manager import (
    notify_log, notify_progress, DownloadUIEvents
)
from smartcash.dataset.services.downloader.notification_utils import (
    notify_service_event, notify_download
)
from smartcash.components.observer import ObserverManager

class TestServiceIntegration(unittest.TestCase):
    """Test suite untuk integrasi antara UI dan service download dataset"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock UI components
        self.ui_components = {
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'output_dir': MagicMock(value='/tmp/test-output'),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='/tmp/test-backup'),
            'log_output': MagicMock(),
            'progress_bar': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'current_progress': MagicMock(),
            'progress_container': MagicMock(layout=MagicMock()),
            'log_accordion': MagicMock(selected_index=0),
            'confirmation_area': MagicMock(clear_output=MagicMock()),
            'status_panel': MagicMock(),
            'download_button': MagicMock(disabled=False),
            'check_button': MagicMock(disabled=False),
            'reset_button': MagicMock(disabled=False),
            'cleanup_button': MagicMock(disabled=False),
            'download_running': False
        }
        
        # Setup mock untuk progress bar dan labels
        for key in ['progress_bar', 'overall_label', 'step_label', 'current_progress']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].value = 0
            self.ui_components[key].description = ""
        
        # Setup mock untuk buttons
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].disabled = False
            
        # Mock observer manager
        self.observer_manager = MagicMock()
        
    @patch('smartcash.dataset.manager.DatasetManager.download_from_roboflow')
    @patch('smartcash.dataset.manager.DatasetManager.get_service')
    @patch('inspect.signature')
    def test_parameter_mapping_from_ui_to_service(self, mock_signature, mock_get_service, mock_download):
        """Test parameter yang dikirim dari UI ke service sudah tepat"""
        # Setup mock service
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        
        # Setup mock download result
        mock_download.return_value = {
            'status': 'success',
            'message': 'Download berhasil'
        }
        
        # Setup mock signature
        mock_param = MagicMock()
        mock_param.default = inspect.Parameter.empty
        mock_signature.return_value = MagicMock(
            parameters={
                'workspace': mock_param,
                'project': mock_param,
                'version': mock_param,
                'api_key': mock_param,
                'output_dir': mock_param,
                'format': MagicMock(default='yolov5pytorch'),
                'show_progress': MagicMock(default=True),
                'verify_integrity': MagicMock(default=True),
                'backup_existing': MagicMock(default=False)
            }
        )
        
        # Patch register_ui_observers
        with patch('smartcash.ui.dataset.download.utils.ui_observers.register_ui_observers') as mock_register:
            mock_register.return_value = self.observer_manager
            
            # Patch os.makedirs untuk mencegah pembuatan direktori sebenarnya
            with patch('os.makedirs'):
                # Patch open untuk mencegah pembuatan file sebenarnya
                with patch('builtins.open', MagicMock()):
                    # Patch os.remove untuk mencegah penghapusan file sebenarnya
                    with patch('os.remove'):
                        # Jalankan fungsi download
                        result = _download_from_roboflow(self.ui_components)
                        
                        # Verifikasi parameter yang dikirim ke download_from_roboflow
                        self.assertTrue(mock_download.called, "download_from_roboflow tidak dipanggil")
                        
                        # Dapatkan parameter yang dikirim
                        call_args, call_kwargs = mock_download.call_args
                        
                        # Verifikasi parameter minimal yang diharapkan
                        expected_params = {
                            'api_key': 'test-api-key',
                            'workspace': 'test-workspace',
                            'project': 'test-project',
                            'version': '1',
                            'output_dir': '/tmp/test-output'
                        }
                        
                        for key, value in expected_params.items():
                            self.assertIn(key, call_kwargs, f"Parameter '{key}' tidak ditemukan")
                            self.assertEqual(call_kwargs[key], value, f"Nilai parameter '{key}' tidak sesuai")
                        
                        # Verifikasi hasil
                        self.assertEqual(result['status'], 'success')
                        self.assertEqual(result['message'], 'Download berhasil')
    
    @patch('smartcash.dataset.manager.DatasetManager.get_service')
    def test_notification_mapping_between_service_and_ui(self, mock_get_service):
        """Test notifikasi yang dikirim antara service dan UI sudah sesuai"""
        # Setup mock service
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service
        
        # Patch register_ui_observers untuk mengembalikan observer manager kita
        with patch('smartcash.ui.dataset.download.utils.ui_observers.register_ui_observers') as mock_register:
            mock_register.return_value = self.observer_manager
            
            # Patch notify_progress untuk menangkap pemanggilan
            with patch('smartcash.ui.dataset.download.handlers.download_handler.notify_progress') as mock_notify_progress:
                # Patch download_from_roboflow untuk mengirim notifikasi
                with patch('smartcash.dataset.manager.DatasetManager.download_from_roboflow') as mock_download:
                    # Setup mock download untuk mengirim notifikasi
                    def mock_download_with_notification(*args, **kwargs):
                        # Kirim notifikasi dari service
                        notify_service_event(
                            "download", 
                            "start", 
                            mock_service, 
                            self.observer_manager,
                            message="Memulai download dataset"
                        )
                        
                        notify_service_event(
                            "download", 
                            "progress", 
                            mock_service, 
                            self.observer_manager,
                            message="Download progress 50%",
                            progress=50,
                            total=100,
                            step="download",
                            current_step=2,
                            total_steps=5
                        )
                        
                        notify_service_event(
                            "download", 
                            "complete", 
                            mock_service, 
                            self.observer_manager,
                            message="Download selesai"
                        )
                        
                        # Return hasil
                        return {
                            'status': 'success',
                            'message': 'Download berhasil'
                        }
                    
                    mock_download.side_effect = mock_download_with_notification
                    
                    # Patch os.makedirs untuk mencegah pembuatan direktori sebenarnya
                    with patch('os.makedirs'):
                        # Patch open untuk mencegah pembuatan file sebenarnya
                        with patch('builtins.open', MagicMock()):
                            # Patch os.remove untuk mencegah penghapusan file sebenarnya
                            with patch('os.remove'):
                                # Jalankan fungsi download
                                result = _download_from_roboflow(self.ui_components)
                                
                                # Verifikasi notify_progress dipanggil
                                self.assertTrue(mock_notify_progress.called, "notify_progress tidak dipanggil")
                                
                                # Verifikasi parameter notifikasi
                                mock_notify_progress.assert_any_call(
                                    sender=self.ui_components,
                                    event_type="start",
                                    progress=0,
                                    total=100,
                                    message="Mempersiapkan download dataset...",
                                    step=1,
                                    total_steps=5
                                )
                                
                                # Verifikasi hasil
                                self.assertEqual(result['status'], 'success')
                                self.assertEqual(result['message'], 'Download berhasil')
    
    def test_parameter_compatibility_between_ui_and_service(self):
        """Test kompatibilitas parameter antara UI dan service download"""
        # Dapatkan signature dari method download_from_roboflow di DatasetManager
        dataset_manager = DatasetManager()
        
        # Patch download_from_roboflow untuk menghindari eksekusi sebenarnya
        with patch.object(dataset_manager, 'download_from_roboflow'):
            # Dapatkan signature dari method download_from_roboflow
            signature = inspect.signature(dataset_manager.download_from_roboflow)
            service_params = list(signature.parameters.keys())
            
            # Parameter yang diharapkan dari UI
            expected_ui_params = [
                'api_key', 'workspace', 'project', 'version', 'output_dir'
            ]
            
            # Verifikasi semua parameter yang diharapkan dari UI ada di service
            for param in expected_ui_params:
                self.assertIn(param, service_params, f"Parameter '{param}' tidak ditemukan di service")

if __name__ == '__main__':
    unittest.main() 
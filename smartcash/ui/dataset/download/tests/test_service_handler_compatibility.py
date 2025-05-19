"""
File: smartcash/ui/dataset/download/tests/test_service_handler_compatibility.py
Deskripsi: Test untuk memastikan kompatibilitas antara service dan handler pada modul download dataset
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets

from smartcash.ui.dataset.download.handlers.download_handler import DownloadHandler
from smartcash.ui.dataset.download.handlers.cleanup_handler import CleanupHandler
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.dataset.manager import DatasetManager

class TestServiceHandlerCompatibility(unittest.TestCase):
    """Test suite untuk memastikan kompatibilitas antara service dan handler."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Buat mock UI components
        self.ui_components = self._create_mock_ui_components()
        
        # Setup logger mock
        self.logger_mock = MagicMock()
        self.ui_components['logger'] = self.logger_mock
        
        # Set output_dir to match test expectation
        self.ui_components['output_dir'].value = 'data/test'
        
        # Setup mock untuk DatasetManager dan DownloadService
        self.dataset_manager_mock = MagicMock()
        self.download_service_mock = MagicMock()
        
        # Setup patch untuk import
        self.dataset_manager_patch = patch('smartcash.dataset.manager.DatasetManager',
                                         return_value=self.dataset_manager_mock)
        self.download_service_patch = patch('smartcash.dataset.services.downloader.download_service.DownloadService',
                                          return_value=self.download_service_mock)
        
        # Start patches
        self.dataset_manager_mock = self.dataset_manager_patch.start()
        self.download_service_mock = self.download_service_patch.start()
        
        # Setup handlers
        self.download_handler = DownloadHandler(ui_components=self.ui_components)
        self.cleanup_handler = CleanupHandler(ui_components=self.ui_components)
        
        # Reset download_running flag
        self.ui_components['download_running'] = False
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        # Stop patches
        self.dataset_manager_patch.stop()
        self.download_service_patch.stop()
        
        # Hapus temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_ui_components(self):
        """Buat mock UI components untuk testing."""
        # Buat mock untuk semua komponen UI yang diperlukan
        ui_components = {
            'download_button': MagicMock(spec=widgets.Button),
            'check_button': MagicMock(spec=widgets.Button),
            'reset_button': MagicMock(spec=widgets.Button),
            'save_button': MagicMock(spec=widgets.Button),
            'source_dropdown': MagicMock(spec=widgets.Dropdown, value='roboflow'),
            'backup_checkbox': MagicMock(spec=widgets.Checkbox, value=True),
            'backup_dir': MagicMock(spec=widgets.Text, value='data/backup'),
            'cleanup_button': MagicMock(spec=widgets.Button),
            'progress_bar': MagicMock(spec=widgets.FloatProgress),
            'overall_label': MagicMock(spec=widgets.HTML),
            'step_label': MagicMock(spec=widgets.HTML),
            'status_panel': MagicMock(spec=widgets.HTML),
            'log_output': MagicMock(spec=widgets.Output),
            'summary_container': MagicMock(spec=widgets.Output),
            'confirmation_area': MagicMock(spec=widgets.Output),
            'progress_container': MagicMock(spec=widgets.VBox),
            'workspace': MagicMock(spec=widgets.Text, value='test-workspace'),
            'project': MagicMock(spec=widgets.Text, value='test-project'),
            'version': MagicMock(spec=widgets.Text, value='1'),
            'api_key': MagicMock(spec=widgets.Text, value='test-api-key'),
            'output_dir': MagicMock(spec=widgets.Text, value='data/test'),
            'validate_dataset': MagicMock(spec=widgets.Checkbox, value=True),
            'reset_progress_bar': MagicMock(),
            'dataset_stats': {
                'total_images': 100,
                'total_labels': 90,
                'classes': {'0': 50, '1': 40}
            },
            'download_timestamp': '2025-05-19T12:14:39+07:00',
            'download_running': False
        }
        
        # Tambahkan layout ke komponen yang membutuhkannya
        for key in ['progress_bar', 'overall_label', 'step_label', 'progress_container', 
                   'summary_container', 'confirmation_area']:
            ui_components[key].layout = MagicMock()
            ui_components[key].layout.visibility = 'visible'
            ui_components[key].layout.display = 'block'
        
        return ui_components
    
    def test_download_handler_compatibility_with_service(self):
        """Test kompatibilitas download handler dengan service."""
        # Setup mock untuk download_from_roboflow
        self.dataset_manager_mock.download_from_roboflow.return_value = True
        
        # Panggil fungsi download
        self.download_handler.download()
        
        # Verifikasi bahwa download_from_roboflow dipanggil dengan parameter yang benar
        self.dataset_manager_mock.download_from_roboflow.assert_called_once_with(
            workspace=self.ui_components['workspace'].value,
            project=self.ui_components['project'].value,
            version=self.ui_components['version'].value,
            api_key=self.ui_components['api_key'].value,
            output_dir=self.ui_components['output_dir'].value,
            validate_dataset=self.ui_components['validate_dataset'].value,
            backup_before_download=self.ui_components['backup_checkbox'].value,
            backup_dir=self.ui_components['backup_dir'].value
        )
    
    def test_cleanup_handler_compatibility_with_service(self):
        """Test kompatibilitas cleanup handler dengan service."""
        # Setup mock untuk cleanup_dataset
        self.dataset_manager_mock.cleanup_dataset.return_value = True
        
        # Panggil fungsi cleanup
        self.cleanup_handler.cleanup()
        
        # Verifikasi bahwa cleanup_dataset dipanggil dengan parameter yang benar
        self.dataset_manager_mock.cleanup_dataset.assert_called_once_with(
            output_dir=self.ui_components['output_dir'].value
        )
    
    def test_download_service_parameter_compatibility(self):
        """Test kompatibilitas parameter antara handler dan service."""
        # Setup mock untuk download_from_roboflow
        self.dataset_manager_mock.download_from_roboflow.return_value = True
        
        # Panggil fungsi download
        self.download_handler.download()
        
        # Verifikasi bahwa parameter yang dikirim ke service sesuai dengan yang diharapkan
        call_args = self.dataset_manager_mock.download_from_roboflow.call_args[1]
        self.assertEqual(call_args['workspace'], 'test-workspace')
        self.assertEqual(call_args['project'], 'test-project')
        self.assertEqual(call_args['version'], '1')
        self.assertEqual(call_args['api_key'], 'test-api-key')
        self.assertEqual(call_args['output_dir'], self.temp_dir)
        self.assertTrue(call_args['validate_dataset'])
        self.assertTrue(call_args['backup_before_download'])
        self.assertEqual(call_args['backup_dir'], 'data/backup')
    
    def test_integration_with_dataset_manager(self):
        """Test integrasi handler dengan DatasetManager."""
        # Setup mock untuk DatasetManager
        self.dataset_manager_mock.download_from_roboflow.return_value = True
        self.dataset_manager_mock.cleanup_dataset.return_value = True
        
        # Test download
        self.download_handler.download()
        self.dataset_manager_mock.download_from_roboflow.assert_called_once()
        
        # Test cleanup
        self.cleanup_handler.cleanup()
        self.dataset_manager_mock.cleanup_dataset.assert_called_once()
    
    def test_error_handling_compatibility(self):
        """Test kompatibilitas penanganan error antara handler dan service."""
        # Setup mock untuk raise exception
        self.dataset_manager_mock.download_from_roboflow.side_effect = Exception("Test error")
        
        # Panggil fungsi download
        self.download_handler.download()
        
        # Verifikasi bahwa error ditangani dengan benar
        self.assertFalse(self.ui_components['download_running'])
        self.ui_components['download_button'].disabled = False
        self.ui_components['check_button'].disabled = False
        self.ui_components['reset_button'].disabled = False
        self.ui_components['save_button'].disabled = False
        self.ui_components['cleanup_button'].disabled = False

if __name__ == '__main__':
    unittest.main()

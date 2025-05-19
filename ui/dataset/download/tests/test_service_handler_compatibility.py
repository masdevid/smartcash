"""
File: smartcash/ui/dataset/download/tests/test_service_handler_compatibility.py
Deskripsi: Test untuk memastikan kesesuaian antara service dan handler pada modul download
"""

import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, call

import ipywidgets as widgets

class TestServiceHandlerCompatibility(unittest.TestCase):
    """Test suite untuk memastikan kesesuaian antara service dan handler."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Buat mock UI components
        self.ui_components = self._create_mock_ui_components()
        
        # Setup logger mock
        self.logger_mock = MagicMock()
        self.ui_components['logger'] = self.logger_mock
        
        # Setup temporary directory untuk test
        self.temp_dir = tempfile.mkdtemp()
        self.ui_components['output_dir'].value = self.temp_dir
        
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
        """Test kesesuaian antara download handler dan service."""
        from smartcash.ui.dataset.download.handlers.download_handler import _download_from_roboflow
        from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
        
        # Tambahkan mock untuk DatasetManager
        dataset_manager = MagicMock()
        dataset_manager.download_from_roboflow.return_value = {
            'success': True,
            'stats': {
                'total_images': 100,
                'total_labels': 90,
                'classes': {'0': 50, '1': 40}
            },
            'elapsed_time': 10.5
        }
        
        # Patch DatasetManager dan method-nya
        with patch('smartcash.dataset.manager.DatasetManager', return_value=dataset_manager), \
             patch('smartcash.ui.dataset.download.handlers.download_handler._process_download_result') as process_mock, \
             patch('smartcash.ui.dataset.download.handlers.endpoint_handler.get_endpoint_config') as config_mock:
            
            # Setup config mock
            config_mock.return_value = {
                'workspace': 'test-workspace',
                'project': 'test-project',
                'version': '1',
                'api_key': 'test-api-key',
                'format': 'yolov5pytorch',
                'output_dir': 'data/test',
                'validate': True
            }
            
            # Panggil fungsi download
            _download_from_roboflow(self.ui_components)
            
            # Verifikasi bahwa dataset_manager.download_from_roboflow dipanggil
            dataset_manager.download_from_roboflow.assert_called_once()
            
            # Dapatkan argumen yang digunakan untuk memanggil download_from_roboflow
            call_args = dataset_manager.download_from_roboflow.call_args
            
            # Verifikasi parameter kunci
            self.assertEqual(call_args[1]['api_key'], 'test-api-key')
            self.assertEqual(call_args[1]['workspace'], 'test-workspace')
            self.assertEqual(call_args[1]['project'], 'test-project')
            self.assertEqual(call_args[1]['version'], '1')
            self.assertEqual(call_args[1]['format'], 'yolov5pytorch')
            self.assertEqual(call_args[1]['output_dir'], 'data/test')
            self.assertEqual(call_args[1]['verify_integrity'], True)
            
            # Verifikasi bahwa _process_download_result dipanggil dengan hasil yang benar
            process_mock.assert_called_once()
    
    def test_cleanup_handler_compatibility_with_service(self):
        """Test kesesuaian antara cleanup handler dan service."""
        from smartcash.ui.dataset.download.handlers.cleanup_handler import cleanup_ui
        from smartcash.ui.dataset.download.handlers.cleanup_button_handler import handle_cleanup_button_click
        
        # Setup mock untuk cleanup_ui
        cleanup_ui_mock = MagicMock()
        self.ui_components['cleanup_ui'] = cleanup_ui_mock
        
        # Panggil handler cleanup button
        with patch('smartcash.ui.dataset.download.handlers.cleanup_button_handler._execute_cleanup') as execute_mock:
            button_mock = MagicMock()
            handle_cleanup_button_click(button_mock, self.ui_components)
            
            # Verifikasi bahwa _execute_cleanup dipanggil
            execute_mock.assert_called_once()
        
        # Test cleanup_ui langsung
        cleanup_ui(self.ui_components)
        
        # Verifikasi bahwa tombol-tombol diaktifkan kembali
        self.assertFalse(self.ui_components['download_button'].disabled)
        self.assertFalse(self.ui_components['check_button'].disabled)
        self.assertFalse(self.ui_components['reset_button'].disabled)
        self.assertFalse(self.ui_components['cleanup_button'].disabled)
        
        # Verifikasi bahwa progress bar direset
        self.assertEqual(self.ui_components['progress_bar'].value, 0)
        self.assertEqual(self.ui_components['progress_bar'].layout.visibility, 'hidden')
        
        # Verifikasi bahwa flag download_running diset ke False
        self.assertFalse(self.ui_components['download_running'])
    
    def test_download_service_parameter_compatibility(self):
        """Test kesesuaian parameter antara download handler dan service."""
        # Patch untuk mendapatkan parameter dari download service
        with patch('smartcash.dataset.services.downloader.download_service.DownloadService') as mock_service, \
             patch('smartcash.ui.dataset.download.handlers.endpoint_handler.get_endpoint_config') as config_mock:
            
            # Setup config mock
            config_mock.return_value = {
                'workspace': 'test-workspace',
                'project': 'test-project',
                'version': '1',
                'api_key': 'test-api-key',
                'format': 'yolov5pytorch',
                'output_dir': 'data/test',
                'validate': True
            }
            
            # Verifikasi bahwa parameter yang digunakan di handler sesuai dengan service
            from smartcash.ui.dataset.download.handlers.download_handler import _download_from_roboflow
            
            # Panggil fungsi download dengan mock
            with patch('smartcash.dataset.manager.DatasetManager') as mock_dataset_manager_class:
                # Setup mock dataset manager
                mock_manager = MagicMock()
                mock_dataset_manager_class.return_value = mock_manager
                mock_manager.download_from_roboflow.return_value = {'success': True}
                
                # Patch _process_download_result untuk menghindari error
                with patch('smartcash.ui.dataset.download.handlers.download_handler._process_download_result'):
                    # Panggil fungsi download
                    _download_from_roboflow(self.ui_components)
                
                # Verifikasi parameter yang digunakan
                call_args = mock_manager.download_from_roboflow.call_args[1]
                
                # Verifikasi parameter kunci yang diharapkan
                expected_params = ['api_key', 'workspace', 'project', 'version', 'format', 'output_dir', 'verify_integrity']
                for param in expected_params:
                    self.assertIn(param, call_args, f"Parameter {param} tidak ditemukan dalam pemanggilan download_from_roboflow")
                
                # Verifikasi konversi validate ke verify_integrity
                self.assertEqual(call_args['verify_integrity'], True)
    
    def test_setup_handlers_integration(self):
        """Test integrasi setup handlers dengan service."""
        from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
        
        # Setup mock untuk fungsi-fungsi yang dipanggil oleh setup_download_handlers
        with patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_observers') as observers_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_api_key_handler') as api_key_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_endpoint_handlers') as endpoint_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_download_button_handler') as download_button_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_check_button_handler') as check_button_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_reset_button_handler') as reset_button_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_save_button_handler') as save_button_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_progress_tracking') as progress_mock, \
             patch('smartcash.ui.dataset.download.handlers.setup_handlers._setup_cleanup') as cleanup_mock:
            
            # Panggil setup_download_handlers
            result = setup_download_handlers(self.ui_components)
            
            # Verifikasi bahwa semua fungsi setup dipanggil
            observers_mock.assert_called_once_with(self.ui_components)
            api_key_mock.assert_called_once_with(self.ui_components)
            endpoint_mock.assert_called_once_with(self.ui_components)
            download_button_mock.assert_called_once_with(self.ui_components)
            check_button_mock.assert_called_once_with(self.ui_components)
            reset_button_mock.assert_called_once_with(self.ui_components)
            save_button_mock.assert_called_once_with(self.ui_components)
            progress_mock.assert_called_once_with(self.ui_components)
            cleanup_mock.assert_called_once_with(self.ui_components)
            
            # Verifikasi bahwa hasil yang dikembalikan adalah ui_components yang sama
            self.assertEqual(result, self.ui_components)
    
    def test_integration_with_dataset_manager(self):
        """Test integrasi dengan DatasetManager."""
        from smartcash.ui.dataset.download.handlers.download_handler import handle_download_button_click
        
        # Setup mock untuk konfirmasi download
        with patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download') as confirm_mock, \
             patch('smartcash.ui.dataset.download.handlers.download_handler._disable_buttons') as disable_mock, \
             patch('smartcash.ui.dataset.download.handlers.download_handler._reset_progress_bar') as reset_mock:
            
            # Panggil handler download button
            button_mock = MagicMock()
            handle_download_button_click(button_mock, self.ui_components)
            
            # Verifikasi bahwa fungsi-fungsi penting dipanggil
            disable_mock.assert_called_once_with(self.ui_components, True)
            reset_mock.assert_called_once_with(self.ui_components)
            confirm_mock.assert_called_once()

if __name__ == '__main__':
    unittest.main()

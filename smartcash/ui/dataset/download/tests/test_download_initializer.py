"""
File: smartcash/ui/dataset/download/tests/test_download_initializer.py
Deskripsi: Test untuk download initializer dengan fokus pada konfigurasi dan notifikasi
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import tempfile
from pathlib import Path

from smartcash.ui.dataset.download import download_initializer

class TestDownloadInitializer(unittest.TestCase):
    """Test untuk download initializer"""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Setup temporary directory
        self.temp_dir = Path("temp_test_download")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup test config
        self.config = {
            'data': {
                'download': {
                    'source': 'roboflow',
                    'output_dir': str(self.temp_dir),
                    'backup_before_download': True,
                    'backup_dir': str(self.temp_dir / 'backups')
                },
                'roboflow': {
                    'workspace': 'smartcash-wo2us',
                    'project': 'rupiah-emisi-2022',
                    'version': '3',
                    'api_key': 'test-api-key'
                }
            }
        }
        # Setup patches
        self.config_manager_patch = patch('smartcash.ui.dataset.download.download_initializer.get_config_manager')
        self.environment_manager_patch = patch('smartcash.ui.dataset.download.download_initializer.get_environment_manager')
        self.create_ui_patch = patch('smartcash.ui.dataset.download.download_initializer.create_download_ui')
        self.observer_patch = patch('smartcash.ui.dataset.download.download_initializer.get_observer_manager')
        self.setup_handlers_patch = patch('smartcash.ui.dataset.download.download_initializer.setup_download_handlers')
        self.load_config_patch = patch('smartcash.ui.dataset.download.download_initializer.load_config')
        self.update_ui_patch = patch('smartcash.ui.dataset.download.download_initializer.update_ui_from_config')
        self.notify_patch = patch('smartcash.ui.dataset.download.download_initializer.notify_service_event')
        # Start patches
        self.config_manager_mock = self.config_manager_patch.start()
        self.environment_manager_mock = self.environment_manager_patch.start()
        self.mock_create_ui = self.create_ui_patch.start()
        self.mock_observer = self.observer_patch.start()
        self.mock_setup_handlers = self.setup_handlers_patch.start()
        self.mock_load_config = self.load_config_patch.start()
        self.mock_update_ui = self.update_ui_patch.start()
        self.mock_notify = self.notify_patch.start()
        # Setup mock returns
        self.config_manager_mock.return_value.get_module_config.return_value = self.config
        self.environment_manager_mock.return_value.base_dir = str(self.temp_dir)
        self.mock_ui_components = { 'logger': MagicMock(), 'output_dir': MagicMock(value=str(self.temp_dir)), 'ui': MagicMock(), 'log_output': MagicMock(), 'progress_container': MagicMock(), 'status_panel': MagicMock() }
        self.mock_create_ui.return_value = self.mock_ui_components
        self.mock_observer_manager = MagicMock()
        self.mock_observer.return_value = self.mock_observer_manager
        self.mock_setup_handlers.return_value = self.mock_ui_components
        self.mock_load_config.return_value = self.config
        # Import after patches
        from smartcash.ui.dataset.download import download_initializer
        self.download_initializer = download_initializer
    def tearDown(self):
        self.config_manager_patch.stop()
        self.environment_manager_patch.stop()
        self.create_ui_patch.stop()
        self.observer_patch.stop()
        self.setup_handlers_patch.stop()
        self.load_config_patch.stop()
        self.update_ui_patch.stop()
        self.notify_patch.stop()

    def test_initialize_dataset_download_ui(self):
        """Test inisialisasi UI download dataset"""
        ui_components = self.download_initializer.initialize_dataset_download_ui()
        
        # Verify observer setup
        self.mock_observer.assert_called_once()
        
        # Verify UI creation and setup
        self.mock_create_ui.assert_called_once()
        self.mock_setup_handlers.assert_called_once_with(self.mock_ui_components, config=None)
        self.mock_load_config.assert_called_once()
        self.mock_update_ui.assert_called_once()
        
        # Verify notification setup
        self.mock_notify.assert_any_call(
            "download",
            "start",
            ANY,
            ANY,
            message="Konfigurasi berhasil dimuat",
            step="config"
        )
        
        # Verify returned components
        self.assertEqual(ui_components, self.mock_ui_components)
        
        # Verify cleanup function is added
        self.assertIn('cleanup', ui_components)
        self.assertTrue(callable(ui_components['cleanup']))

    def test_error_handling(self):
        """Test error handling saat inisialisasi"""
        # Simulate error in create_download_ui
        self.mock_create_ui.side_effect = Exception("Test error")
        
        # Verify that error is logged and re-raised
        with self.assertRaises(Exception):
            self.download_initializer.initialize_dataset_download_ui()
        
        # Verify logger was called (fallback logger)
        self.mock_ui_components['logger'].error.assert_called()
        
        # Verify error notification
        self.mock_notify.assert_any_call(
            "download",
            "error",
            ANY,
            ANY,
            message="Gagal memuat konfigurasi: Test error",
            step="config"
        )

    def test_cleanup_function(self):
        """Test cleanup function"""
        ui_components = self.download_initializer.initialize_dataset_download_ui()
        
        # Call cleanup function
        ui_components['cleanup']()
        
        # Verify observer cleanup (unregister_all)
        self.mock_observer_manager.unregister_all.assert_called_once()
        
        # Verify UI reset
        self.assertEqual(ui_components['progress_container'].layout.display, 'none')
        self.assertEqual(ui_components['status_panel'].value, "Siap untuk download dataset")
        ui_components['log_output'].clear_output.assert_called_once()
        
        # Verify cleanup notification
        self.mock_notify.assert_any_call(
            "download",
            "progress",
            ANY,
            ANY,
            message="Membersihkan resources...",
            step="cleanup"
        )
        self.mock_notify.assert_any_call(
            "download",
            "complete",
            ANY,
            ANY,
            message="Resources berhasil dibersihkan",
            step="cleanup"
        )

if __name__ == '__main__':
    unittest.main() 
"""
File: smartcash/ui/dataset/download/tests/test_download_initializer.py
Deskripsi: Test untuk download initializer dengan fokus pada konfigurasi dan notifikasi
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from smartcash.ui.dataset.download import download_initializer

class TestDownloadInitializer(unittest.TestCase):
    """Test untuk download initializer"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_observer_manager = MagicMock()
        self.mock_observer_manager.unregister_all = MagicMock()
        self.mock_ui_components = {
            'ui': MagicMock(),
            'progress_bar': MagicMock(),
            'logger': MagicMock(),
            'config': {},
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'output_dir': MagicMock(value='data/test'),
            'validate_dataset': MagicMock(value=True),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='data/backup'),
            'status_panel': MagicMock(),
            'log_output': MagicMock(),
            'progress_container': MagicMock(),
            'action_buttons': {
                'primary_button': MagicMock(),
                'secondary_buttons': [MagicMock()]
            }
        }

        # Patch where used in download_initializer
        self.patcher_observer = patch('smartcash.components.observer.get_observer_manager', 
                                    return_value=self.mock_observer_manager)
        self.patcher_create_ui = patch('smartcash.ui.dataset.download.download_initializer.create_download_ui', 
                                     return_value=self.mock_ui_components)
        self.patcher_setup_handlers = patch('smartcash.ui.dataset.download.download_initializer.setup_download_handlers', 
                                          return_value=self.mock_ui_components)
        self.patcher_load_config = patch('smartcash.ui.dataset.download.download_initializer.load_config', 
                                       return_value={})
        self.patcher_update_ui = patch('smartcash.ui.dataset.download.download_initializer.update_ui_from_config')
        self.patcher_notify = patch('smartcash.dataset.services.downloader.notification_utils.notify_service_event')

        self.mock_observer = self.patcher_observer.start()
        self.mock_create_ui = self.patcher_create_ui.start()
        self.mock_setup_handlers = self.patcher_setup_handlers.start()
        self.mock_load_config = self.patcher_load_config.start()
        self.mock_update_ui = self.patcher_update_ui.start()
        self.mock_notify = self.patcher_notify.start()

    def tearDown(self):
        self.patcher_observer.stop()
        self.patcher_create_ui.stop()
        self.patcher_setup_handlers.stop()
        self.patcher_load_config.stop()
        self.patcher_update_ui.stop()
        self.patcher_notify.stop()

    def test_initialize_dataset_download_ui(self):
        """Test inisialisasi UI download dataset"""
        ui_components = download_initializer.initialize_dataset_download_ui()
        
        # Verify observer setup
        self.mock_observer.assert_called_once()
        
        # Verify UI creation and setup
        self.mock_create_ui.assert_called_once()
        self.mock_setup_handlers.assert_called_once_with(self.mock_ui_components)
        self.mock_load_config.assert_called_once()
        self.mock_update_ui.assert_called_once()
        
        # Verify notification setup
        self.mock_notify.assert_called_with(
            "download",
            "start",
            self.mock_ui_components,
            self.mock_observer_manager,
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
            download_initializer.initialize_dataset_download_ui()
        
        # Verify logger was called (either on ui_components or fallback logger)
        logger_mock = self.mock_ui_components['logger']
        self.assertTrue(logger_mock.error.called)
        
        # Verify error notification
        self.mock_notify.assert_called_with(
            "download",
            "error",
            self.mock_ui_components,
            self.mock_observer_manager,
            message="Gagal memuat konfigurasi: Test error",
            step="config"
        )

    def test_cleanup_function(self):
        """Test cleanup function"""
        ui_components = download_initializer.initialize_dataset_download_ui()
        
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
            ui_components,
            self.mock_observer_manager,
            message="Membersihkan resources...",
            step="cleanup"
        )
        
        self.mock_notify.assert_any_call(
            "download",
            "complete",
            ui_components,
            self.mock_observer_manager,
            message="Resources berhasil dibersihkan",
            step="cleanup"
        )

if __name__ == '__main__':
    unittest.main() 
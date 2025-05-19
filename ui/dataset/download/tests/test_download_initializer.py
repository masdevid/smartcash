"""
File: smartcash/ui/dataset/download/tests/test_download_initializer.py
Deskripsi: Test untuk download initializer dengan fokus pada konfigurasi dan notifikasi
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import tempfile
from pathlib import Path
import ipywidgets as widgets
import shutil

class TestDownloadInitializer(unittest.TestCase):
    """Test untuk download initializer"""
    
    @classmethod
    def setUpClass(cls):
        """Setup untuk seluruh test class."""
        # Setup temporary directory
        cls.temp_dir = Path("temp_test_download")
        cls.temp_dir.mkdir(exist_ok=True)
        
        # Setup test config
        cls.config = {
            'data': {
                'download': {
                    'source': 'roboflow',
                    'output_dir': str(cls.temp_dir),
                    'backup_before_download': True,
                    'backup_dir': str(cls.temp_dir / 'backups')
                },
                'roboflow': {
                    'workspace': 'smartcash-wo2us',
                    'project': 'rupiah-emisi-2022',
                    'version': '3',
                    'api_key': 'test-api-key'
                }
            }
        }
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup setelah seluruh test selesai."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Create mock UI components first
        self.mock_ui_components = {
            'logger': MagicMock(),
            'output_dir': MagicMock(value=str(self.temp_dir)),
            'ui': widgets.VBox([]),
            'log_output': MagicMock(),
            'progress_container': MagicMock(),
            'status_panel': MagicMock()
        }

        # Setup patches with proper order to avoid recursion
        self.patches = []
        
        # Import after patches
        from smartcash.ui.dataset.download import download_initializer
        self.download_initializer = download_initializer

    def tearDown(self):
        """Cleanup after each test."""
        # Stop all patches
        for p in self.patches:
            p.stop()
        
        # Clear mock UI components
        self.mock_ui_components.clear()

    def _setup_patches(self):
        """Setup patches untuk test."""
        patches = [
            patch('smartcash.ui.dataset.download.download_initializer.get_config_manager'),
            patch('smartcash.ui.dataset.download.download_initializer.get_environment_manager'),
            patch('smartcash.ui.dataset.download.download_initializer.create_download_ui'),
            patch('smartcash.ui.dataset.download.download_initializer.get_observer_manager'),
            patch('smartcash.ui.dataset.download.download_initializer.setup_download_handlers'),
            patch('smartcash.ui.dataset.download.download_initializer.load_config'),
            patch('smartcash.ui.dataset.download.download_initializer.update_ui_from_config'),
            patch('smartcash.ui.dataset.download.download_initializer.notify_service_event'),
            patch('smartcash.dataset.services.downloader.download_service.DownloadService'),
            patch('smartcash.dataset.services.downloader.backup_service.BackupService'),
            patch('smartcash.common.logger.get_logger')
        ]

        # Start all patches
        self.patches = [p.start() for p in patches]
        
        # Setup mock returns
        self.patches[0].return_value.get_module_config.return_value = self.config
        self.patches[1].return_value.base_dir = str(self.temp_dir)
        self.patches[2].return_value = self.mock_ui_components
        self.mock_observer_manager = MagicMock()
        self.patches[3].return_value = self.mock_observer_manager
        self.patches[4].return_value = self.mock_ui_components
        self.patches[5].return_value = self.config
        self.patches[8].return_value = MagicMock()  # DownloadService mock
        self.patches[9].return_value = MagicMock()  # BackupService mock
        self.patches[10].return_value = MagicMock()  # Logger mock

    def test_initialize_dataset_download_ui(self):
        """Test inisialisasi UI download dataset"""
        self._setup_patches()
        ui_components = self.download_initializer.initialize_dataset_download_ui()
        
        # Verify observer setup
        self.patches[3].assert_called_once()
        
        # Verify UI creation and setup
        self.patches[2].assert_called_once()
        self.patches[4].assert_called_once_with(self.mock_ui_components, config=None)
        self.patches[5].assert_called_once()
        self.patches[6].assert_called_once()
        
        # Verify notification setup
        self.patches[7].assert_any_call(
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
        self._setup_patches()
        # Simulate error in create_download_ui
        self.patches[2].side_effect = Exception("Test error")
        
        # Verify that error is logged and re-raised
        with self.assertRaises(Exception):
            self.download_initializer.initialize_dataset_download_ui()
        
        # Verify logger was called
        self.mock_ui_components['logger'].error.assert_called()
        
        # Verify error notification
        self.patches[7].assert_any_call(
            "download",
            "error",
            ANY,
            ANY,
            message="Gagal memuat konfigurasi: Test error",
            step="config"
        )

    def test_cleanup_function(self):
        """Test cleanup function"""
        self._setup_patches()
        ui_components = self.download_initializer.initialize_dataset_download_ui()
        
        # Call cleanup function
        ui_components['cleanup']()
        
        # Verify observer cleanup
        self.mock_observer_manager.unregister_all.assert_called_once()
        
        # Verify UI reset
        self.assertEqual(ui_components['progress_container'].layout.display, 'none')
        self.assertEqual(ui_components['status_panel'].value, "Siap untuk download dataset")
        ui_components['log_output'].clear_output.assert_called_once()
        
        # Verify cleanup notification
        self.patches[7].assert_any_call(
            "download",
            "progress",
            ANY,
            ANY,
            message="Membersihkan resources...",
            step="cleanup"
        )
        self.patches[7].assert_any_call(
            "download",
            "complete",
            ANY,
            ANY,
            message="Resources berhasil dibersihkan",
            step="cleanup"
        )

if __name__ == '__main__':
    unittest.main() 
"""
File: tests/ui/dataset/downloader/test_downloader_init.py
Deskripsi: Test untuk downloader_init.py dan integrasinya
"""

import unittest
from unittest.mock import MagicMock, patch, ANY, mock_open
import ipywidgets as widgets
from pathlib import Path
import yaml

# Import modul yang akan di-test
from smartcash.ui.dataset.downloader.downloader_init import DownloaderInitializer, DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.setup_handlers import setup_download_handlers

class TestDownloaderInitializer(unittest.TestCase):
    """Test case untuk DownloaderInitializer."""
    
    @classmethod
    def setUpClass(cls):
        """Setup class untuk test."""
        cls.config = {
            'roboflow': {
                'workspace': 'test-workspace',
                'project': 'test-project',
                'version': '1',
                'api_key': 'test-api-key',
                'output_format': 'yolov5pytorch'
            },
            'data': {
                'dir': 'test-data',
                'preprocessed_dir': 'test-data/preprocessed'
            },
            'cleanup': {
                'backup_dir': 'test-backup',
                'backup_enabled': True
            }
        }
    
    def setUp(self):
        """Setup untuk setiap test case."""
        self.mock_ui = {
            'workspace_field': widgets.Text(value='test-workspace'),
            'project_field': widgets.Text(value='test-project'),
            'version_field': widgets.Text(value='1'),
            'api_key_field': widgets.Password(value='test-api-key'),
            'output_dir_field': widgets.Text(value='test-data'),
            'backup_dir_field': widgets.Text(value='test-backup'),
            'backup_checkbox': widgets.Checkbox(value=True),
            'organize_dataset': widgets.Checkbox(value=True),
            'download_button': widgets.Button(description='Download'),
            'check_button': widgets.Button(description='Check'),
            'cleanup_button': widgets.Button(description='Cleanup'),
            'status_panel': widgets.Output(),
            'log_output': widgets.Output(),
            'progress_bar': widgets.FloatProgress(value=0.0, min=0.0, max=1.0)
        }
        
        # Mock logger
        self.mock_logger = MagicMock()
        
        # Inisialisasi DownloaderInitializer
        self.initializer = DownloaderInitializer(
            ui_components=self.mock_ui,
            env=None,
            logger=self.mock_logger
        )
    
    def test_init(self):
        """Test inisialisasi DownloaderInitializer."""
        self.assertIsNotNone(self.initializer)
        self.assertEqual(self.initializer.ui_components, self.mock_ui)
        self.assertEqual(self.initializer.logger, self.mock_logger)
    
    @patch('smartcash.ui.dataset.downloader.handlers.setup_handlers.setup_download_handlers')
    def test_initialize(self, mock_setup_handlers):
        """Test inisialisasi downloader."""
        # Setup mock
        mock_setup_handlers.return_value = {'success': True}
        
        # Panggil method initialize
        result = self.initializer.initialize()
        
        # Verifikasi
        self.assertTrue(result)
        mock_setup_handlers.assert_called_once_with(self.mock_ui, ANY)
        self.mock_logger.info.assert_called_with("✅ Downloader initialized successfully")
    
    @patch('smartcash.ui.dataset.downloader.handlers.setup_handlers.setup_download_handlers')
    def test_initialize_failure(self, mock_setup_handlers):
        """Test inisialisasi downloader gagal."""
        # Setup mock untuk gagal
        mock_setup_handlers.return_value = {'success': False, 'error': 'Test error'}
        
        # Panggil method initialize
        result = self.initializer.initialize()
        
        # Verifikasi
        self.assertFalse(result)
        mock_setup_handlers.assert_called_once_with(self.mock_ui, ANY)
        self.mock_logger.error.assert_called_with("❌ Failed to initialize downloader: Test error")
    
    @patch('smartcash.ui.dataset.downloader.handlers.setup_handlers.setup_download_handlers')
    @patch('smartcash.ui.dataset.downloader.components.ui_layout.create_downloader_ui')
    def test_initialize_ui(self, mock_create_ui, mock_setup_handlers):
        """Test inisialisasi UI dan setup handlers."""
        # Setup mocks
        mock_ui = {
            'workspace_field': MagicMock(),
            'project_field': MagicMock(),
            'version_field': MagicMock(),
            'api_key_field': MagicMock(),
            'download_button': MagicMock(),
            'validate_button': MagicMock(),
            'reset_button': MagicMock(),
            'status_panel': MagicMock(),
            'log_output': MagicMock()
        }
        mock_create_ui.return_value = mock_ui
        
        # Panggil method yang akan di-test
        initializer = DownloaderInitializer()
        result = initializer.initialize_ui()
        
        # Verifikasi
        mock_create_ui.assert_called_once()
        mock_setup_handlers.assert_called_once_with(
            mock_ui, ANY, env=initializer.env
        )
        self.assertEqual(result, mock_ui)
    
    def test_get_current_config(self):
        """Test mendapatkan konfigurasi saat ini dari UI."""
        # Panggil method
        config = self.initializer.get_current_config()
        
        # Verifikasi
        self.assertEqual(config['roboflow']['workspace'], 'test-workspace')
        self.assertEqual(config['roboflow']['project'], 'test-project')
        self.assertEqual(config['roboflow']['version'], '1')
        self.assertEqual(config['roboflow']['api_key'], 'test-api-key')
        self.assertEqual(config['data']['dir'], 'test-data')
        self.assertEqual(config['cleanup']['backup_dir'], 'test-backup')
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    def test_save_config(self, mock_yaml_dump, mock_file):
        """Test menyimpan konfigurasi."""
        # Panggil method
        self.initializer.save_config('test_config.yaml')
        
        # Verifikasi
        mock_file.assert_called_once_with('test_config.yaml', 'w')
        mock_yaml_dump.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data=yaml.dump({
        'roboflow': {'workspace': 'test'}
    }))
    @patch('yaml.safe_load')
    def test_load_config(self, mock_yaml_load, mock_file):
        """Test memuat konfigurasi."""
        # Setup mock
        mock_yaml_load.return_value = self.config
        
        # Panggil method
        loaded_config = self.initializer.load_config('test_config.yaml')
        
        # Verifikasi
        self.assertEqual(loaded_config, self.config)
        mock_file.assert_called_once_with('test_config.yaml', 'r')
    
    def test_validate_config(self):
        """Test validasi konfigurasi."""
        # Test valid config
        is_valid, errors = self.initializer.validate_config(self.config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid config
        invalid_config = self.config.copy()
        invalid_config['roboflow']['workspace'] = ''
        is_valid, errors = self.initializer.validate_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn('roboflow.workspace', errors[0]['field'])


class TestDownloaderConfigHandler(unittest.TestCase):
    """Test case untuk DownloaderConfigHandler."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        self.handler = DownloaderConfigHandler('test_module')
        self.mock_ui = {
            'workspace_field': MagicMock(value='test-workspace'),
            'project_field': MagicMock(value='test-project'),
            'version_field': MagicMock(value='1'),
            'api_key_field': MagicMock(value='test-api-key'),
            'output_dir_field': MagicMock(value='test-data'),
            'backup_dir_field': MagicMock(value='test-backup'),
            'backup_checkbox': MagicMock(value=True),
            'organize_dataset': MagicMock(value=True)
        }
    
    @patch('smartcash.ui.dataset.downloader.handlers.config_extractor.DownloaderConfigExtractor.extract_config')
    def test_extract_config(self, mock_extract):
        """Test ekstraksi konfigurasi dari UI."""
        # Setup mock
        expected_config = {
            'roboflow': {
                'workspace': 'test-workspace',
                'project': 'test-project',
                'version': '1',
                'api_key': 'test-api-key',
                'output_format': 'yolov5pytorch'
            }
        }
        mock_extract.return_value = expected_config
        
        handler = DownloaderConfigHandler()
        config = handler.extract_config(self.mock_ui)
        
        mock_extract.assert_called_once_with(self.mock_ui)
        self.assertEqual(config, expected_config)
    
    @patch('smartcash.ui.dataset.downloader.handlers.config_updater.DownloaderConfigUpdater.update_ui')
    def test_update_ui(self, mock_update_ui):
        """Test update UI dari konfigurasi."""
        test_config = {
            'roboflow': {
                'workspace': 'new-workspace',
                'project': 'new-project',
                'version': '2',
                'api_key': 'new-api-key'
            }
        }
        self.handler.update_ui(self.mock_ui, test_config)
        mock_update_ui.assert_called_once_with(self.mock_ui, test_config)
    
    @patch('smartcash.ui.dataset.downloader.handlers.defaults.get_default_downloader_config')
    def test_get_default_config(self, mock_get_default):
        """Test mendapatkan konfigurasi default."""
        mock_get_default.return_value = {'test': 'config'}
        config = self.handler.get_default_config()
        self.assertEqual(config, {'test': 'config'})


if __name__ == '__main__':
    unittest.main()

"""
File: smartcash/ui/dataset/download/tests/test_download_initializer.py
Deskripsi: Test untuk download initializer dengan fokus pada konfigurasi
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
        self.mock_env_manager = MagicMock()
        self.mock_env_manager.base_dir = str(self.test_dir)
        self.mock_config_manager = MagicMock()
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
            'progress_container': MagicMock()
        }

        # Patch where used in download_initializer
        self.patcher_env = patch('smartcash.ui.dataset.download.download_initializer.get_environment_manager', return_value=self.mock_env_manager)
        self.patcher_config = patch('smartcash.ui.dataset.download.download_initializer.get_config_manager', return_value=self.mock_config_manager)
        self.patcher_create_ui = patch('smartcash.ui.dataset.download.download_initializer.create_download_ui', return_value=self.mock_ui_components)
        self.patcher_setup_handlers = patch('smartcash.ui.dataset.download.download_initializer.setup_download_handlers', return_value=self.mock_ui_components)
        self.patcher_load_config = patch('smartcash.ui.dataset.download.download_initializer.load_config', return_value={})
        self.patcher_update_ui = patch('smartcash.ui.dataset.download.download_initializer.update_ui_from_config')

        self.mock_env = self.patcher_env.start()
        self.mock_config = self.patcher_config.start()
        self.mock_create_ui = self.patcher_create_ui.start()
        self.mock_setup_handlers = self.patcher_setup_handlers.start()
        self.mock_load_config = self.patcher_load_config.start()
        self.mock_update_ui = self.patcher_update_ui.start()

    def tearDown(self):
        self.patcher_env.stop()
        self.patcher_config.stop()
        self.patcher_create_ui.stop()
        self.patcher_setup_handlers.stop()
        self.patcher_load_config.stop()
        self.patcher_update_ui.stop()

    def test_initialize_dataset_download_ui(self):
        """Test inisialisasi UI download dataset"""
        ui_components = download_initializer.initialize_dataset_download_ui()
        
        # Verify environment setup
        self.mock_env.assert_called_once()
        self.mock_config.assert_called_once_with(
            base_dir=str(self.test_dir),
            config_file='dataset_config.yaml'
        )
        
        # Verify UI creation and setup
        self.mock_create_ui.assert_called_once()
        self.mock_setup_handlers.assert_called_once_with(self.mock_ui_components)
        self.mock_load_config.assert_called_once()
        self.mock_update_ui.assert_called_once()
        
        # Verify config manager registration
        self.mock_config_manager.register_ui_components.assert_called_once_with(
            'dataset_download',
            self.mock_ui_components
        )
        
        # Verify returned components
        self.assertEqual(ui_components, self.mock_ui_components)

    def test_error_handling(self):
        """Test error handling saat inisialisasi"""
        # Simulate error in create_download_ui
        self.mock_create_ui.side_effect = Exception("Test error")
        
        # Verify that error is logged and re-raised
        with self.assertRaises(Exception):
            download_initializer.initialize_dataset_download_ui()
        
        # Verify logger was called
        self.mock_ui_components['logger'].error.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
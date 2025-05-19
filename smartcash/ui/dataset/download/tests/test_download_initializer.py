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
        self.mock_progress_bar = MagicMock()
        self.mock_logger = MagicMock()
        self.mock_ui = MagicMock()
        self.mock_config = {}

        # Patch where used in download_initializer
        self.patcher_env = patch('smartcash.ui.dataset.download.download_initializer.get_environment_manager', return_value=self.mock_env_manager)
        self.patcher_config = patch('smartcash.ui.dataset.download.download_initializer.get_config_manager', return_value=self.mock_config_manager)
        self.patcher_base_init = patch('smartcash.ui.dataset.download.download_initializer.initialize_module_ui', return_value={
            'ui': self.mock_ui,
            'progress_bar': self.mock_progress_bar,
            'logger': self.mock_logger,
            'config': self.mock_config
        })
        self.patcher_colab = patch('smartcash.ui.dataset.download.download_initializer.check_colab_secrets')

        self.mock_env = self.patcher_env.start()
        self.mock_config = self.patcher_config.start()
        self.mock_base_init = self.patcher_base_init.start()
        self.mock_colab = self.patcher_colab.start()

    def tearDown(self):
        self.patcher_env.stop()
        self.patcher_config.stop()
        self.patcher_base_init.stop()
        self.patcher_colab.stop()

    def test_initialize_dataset_download_ui(self):
        ui_components = download_initializer.initialize_dataset_download_ui()
        self.assertIn('ui', ui_components)
        self.assertIn('progress_bar', ui_components)
        self.assertIn('logger', ui_components)
        self.assertIn('config', ui_components)
        self.assertIn('config_manager', ui_components)
        self.assertIs(ui_components['config_manager'], self.mock_config_manager)
        self.mock_base_init.assert_called_once()
        self.mock_env.assert_called_once()
        self.mock_config.assert_called_once_with(base_dir=str(self.test_dir))
        self.mock_colab.assert_called_once()

    def test_config_sync(self):
        ui_components = download_initializer.initialize_dataset_download_ui()
        self.mock_config.assert_called_once_with(base_dir=str(self.test_dir))
        self.assertIn('config_manager', ui_components)
        self.assertIs(ui_components['config_manager'], self.mock_config_manager)

    def test_environment_sync(self):
        ui_components = download_initializer.initialize_dataset_download_ui()
        self.mock_env.assert_called_once()
        self.assertEqual(self.mock_env_manager.base_dir, str(self.test_dir))

if __name__ == '__main__':
    unittest.main() 
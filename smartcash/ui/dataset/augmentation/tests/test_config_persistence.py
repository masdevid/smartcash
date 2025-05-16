"""
File: smartcash/ui/dataset/augmentation/tests/test_config_persistence.py
Deskripsi: Pengujian untuk persistensi konfigurasi augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import os
import yaml

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.config_persistence import (
    ensure_ui_persistence,
    get_augmentation_config,
    save_augmentation_config
)

class TestConfigPersistence(unittest.TestCase):
    """Kelas pengujian untuk config_persistence.py"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Mock komponen UI
        self.mock_ui_components = {
            'status': MagicMock(),
            'logger': MagicMock(),
            'config': {
                'augmentation': {
                    'types': ['Combined (Recommended)'],
                    'prefix': 'aug_',
                    'factor': '2',
                    'split': 'train',
                    'balance_classes': False,
                    'num_workers': 4
                },
                'data': {
                    'dataset_path': '/path/to/dataset'
                }
            }
        }
        
        # Mock ConfigManager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.return_value = self.mock_ui_components['config']
        self.mock_config_manager.save_module_config.return_value = True

    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager')
    def test_ensure_ui_persistence(self, mock_get_manager):
        """Pengujian ensure_ui_persistence"""
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang diuji
        ensure_ui_persistence(self.mock_ui_components)
        
        # Verifikasi hasil
        self.mock_config_manager.register_ui_components.assert_called_once_with('augmentation', self.mock_ui_components)

    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager')
    def test_get_augmentation_config(self, mock_get_manager):
        """Pengujian get_augmentation_config"""
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang diuji
        result = get_augmentation_config()
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components['config'])
        self.mock_config_manager.get_module_config.assert_called_once_with('augmentation', None)

    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager')
    def test_save_augmentation_config(self, mock_get_manager):
        """Pengujian save_augmentation_config"""
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang diuji
        result = save_augmentation_config(self.mock_ui_components['config'])
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called_once_with(
            'augmentation',
            self.mock_ui_components['config']
        )

    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager')
    def test_save_augmentation_config_failure(self, mock_get_manager):
        """Pengujian save_augmentation_config dengan kegagalan"""
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        self.mock_config_manager.save_module_config.return_value = False
        
        # Panggil fungsi yang diuji
        result = save_augmentation_config(self.mock_ui_components['config'])
        
        # Verifikasi hasil
        self.assertFalse(result)
        self.mock_config_manager.save_module_config.assert_called_once_with(
            'augmentation',
            self.mock_ui_components['config']
        )

if __name__ == '__main__':
    unittest.main()

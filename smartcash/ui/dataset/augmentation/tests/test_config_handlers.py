"""
File: smartcash/ui/dataset/augmentation/tests/test_config_handlers.py
Deskripsi: Pengujian untuk handler konfigurasi augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import yaml
import ipywidgets as widgets
from pathlib import Path

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.config_handlers import (
    setup_augmentation_config_handler,
    update_config_from_ui,
    load_augmentation_config,
    update_ui_from_config
)

class TestConfigHandlers(unittest.TestCase):
    """Kelas pengujian untuk config_handlers.py"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Mock komponen UI
        self.mock_ui_components = {
            'aug_options': widgets.VBox([
                widgets.Dropdown(options=['Combined (Recommended)', 'Geometric', 'Color', 'Noise'], value='Combined (Recommended)'),  # types
                widgets.Text(value='aug_'),  # prefix
                widgets.Text(value='2'),  # factor
                widgets.Dropdown(options=['train', 'validation', 'test'], value='train'),  # split
                widgets.Checkbox(value=False),  # balance_classes
                widgets.IntText(value=4)  # num_workers
            ]),
            'status': MagicMock(),
            'logger': MagicMock()
        }
        
        # Mock konfigurasi
        self.mock_config = {
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

    @patch('smartcash.ui.dataset.augmentation.handlers.config_handlers.load_augmentation_config')
    def test_setup_augmentation_config_handler(self, mock_load):
        """Pengujian setup_augmentation_config_handler"""
        # Setup mock
        mock_load.return_value = self.mock_config
        
        # Panggil fungsi yang diuji
        result = setup_augmentation_config_handler(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['config'], self.mock_config)
        mock_load.assert_called_once()

    def test_update_config_from_ui(self):
        """Pengujian update_config_from_ui"""
        # Panggil fungsi yang diuji
        result = update_config_from_ui(self.mock_ui_components, {})
        
        # Verifikasi hasil minimal
        self.assertIsInstance(result, dict)
        self.assertIn('augmentation', result)

    def test_load_augmentation_config(self):
        """Pengujian load_augmentation_config"""
        # Panggil fungsi yang diuji
        result = load_augmentation_config()
        
        # Verifikasi hasil minimal
        self.assertIsInstance(result, dict)
        self.assertIn('augmentation', result)

    def test_update_ui_from_config(self):
        """Pengujian update_ui_from_config"""
        # Setup konfigurasi khusus
        config = {
            'augmentation': {
                'prefix': 'custom_',
                'factor': '5',
                'balance_classes': True,
                'num_workers': 8
            }
        }
        
        # Panggil fungsi yang diuji
        result = update_ui_from_config(self.mock_ui_components, config)
        
        # Verifikasi hasil
        aug_options = result['aug_options'].children
        self.assertEqual(aug_options[1].value, 'custom_')
        self.assertEqual(aug_options[2].value, '5')
        self.assertEqual(aug_options[4].value, True)
        self.assertEqual(aug_options[5].value, 8)

if __name__ == '__main__':
    unittest.main()

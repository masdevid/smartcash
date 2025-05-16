"""
File: smartcash/ui/dataset/augmentation/tests/test_state_handler.py
Deskripsi: Pengujian untuk state handler augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import os
import ipywidgets as widgets

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.state_handler import (
    detect_augmentation_state,
    generate_augmentation_summary
)

class TestStateHandler(unittest.TestCase):
    """Kelas pengujian untuk state_handler.py"""
    
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
            },
            'aug_options': widgets.VBox([
                widgets.Dropdown(options=['Combined (Recommended)', 'Geometric', 'Color', 'Noise'], value='Combined (Recommended)'),
                widgets.Text(value='aug_'),
                widgets.Text(value='2'),
                widgets.Dropdown(options=['train', 'validation', 'test'], value='train'),
                widgets.Checkbox(value=False),
                widgets.IntText(value=4)
            ]),
            'augment_button': MagicMock(),
            'cleanup_button': MagicMock(),
            'summary_container': MagicMock(),
            'is_augmented': False
        }

    @patch('os.path.exists')
    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.ensure_ui_persistence')
    def test_detect_augmentation_state_no_augmentation(self, mock_ensure_persistence, mock_exists):
        """Pengujian detect_augmentation_state tanpa augmentasi sebelumnya"""
        # Setup mock
        mock_exists.return_value = False
        mock_ensure_persistence.return_value = self.mock_ui_components
        
        # Panggil fungsi yang diuji
        result = detect_augmentation_state(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.assertFalse(result.get('is_augmented', True))

    @patch('os.path.exists')
    @patch('pathlib.Path.glob')
    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.ensure_ui_persistence')
    def test_detect_augmentation_state_with_augmentation(self, mock_ensure_persistence, mock_glob, mock_exists):
        """Pengujian detect_augmentation_state dengan augmentasi sebelumnya"""
        # Setup mock
        mock_exists.return_value = True
        mock_glob.return_value = ['aug_img1.jpg', 'aug_img2.jpg']
        mock_ensure_persistence.return_value = self.mock_ui_components
        
        # Panggil fungsi yang diuji
        result = detect_augmentation_state(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)

    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.exists')
    def test_generate_augmentation_summary(self, mock_exists, mock_glob):
        """Pengujian generate_augmentation_summary"""
        # Setup mock
        mock_exists.return_value = True
        mock_glob.return_value = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        
        # Tambahkan komponen yang diperlukan
        self.mock_ui_components['summary_container'] = widgets.Output()
        
        # Panggil fungsi yang diuji
        generate_augmentation_summary(
            self.mock_ui_components,
            preprocessed_dir='/path/to/preprocessed',
            augmented_dir='/path/to/augmented'
        )
        
        # Verifikasi logger dipanggil
        self.mock_ui_components['logger'].warning.assert_not_called()

if __name__ == '__main__':
    unittest.main()

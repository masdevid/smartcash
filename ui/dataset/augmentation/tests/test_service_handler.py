"""
File: smartcash/ui/dataset/augmentation/tests/test_service_handler.py
Deskripsi: Pengujian untuk service handler augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import threading
import os

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import (
    get_augmentation_service,
    register_progress_callback,
    execute_augmentation,
    stop_augmentation
)

class TestAugmentationServiceHandler(unittest.TestCase):
    """Kelas pengujian untuk augmentation_service_handler.py"""
    
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
            'progress_bar': MagicMock(),
            'current_progress': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'output': MagicMock(),
            'augmentation_step': MagicMock(),
            'state': {
                'running': True,
                'completed': False,
                'stop_requested': False
            }
        }
        
        # Mock dataset
        self.mock_dataset = {
            'train': [
                {'image_path': '/path/to/dataset/train/img1.jpg', 'label': 'class1'},
                {'image_path': '/path/to/dataset/train/img2.jpg', 'label': 'class2'}
            ]
        }

    @patch('smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler.AugmentationService')
    def test_get_augmentation_service(self, mock_service_class):
        """Pengujian get_augmentation_service"""
        # Setup mock
        mock_service_instance = MagicMock()
        mock_service_class.return_value = mock_service_instance
        
        # Patch get_augmentation_config yang digunakan dalam fungsi
        with patch('smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler.get_augmentation_config') as mock_get_config:
            mock_get_config.return_value = self.mock_ui_components['config']
            
            # Panggil fungsi yang diuji
            result = get_augmentation_service(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result, mock_service_instance)
            mock_service_class.assert_called_once()
            mock_get_config.assert_called_once_with(self.mock_ui_components)

    def test_register_progress_callback(self):
        """Pengujian register_progress_callback"""
        # Setup mock
        mock_service = MagicMock()
        mock_callback = MagicMock()
        
        # Panggil fungsi yang diuji
        register_progress_callback(mock_service, mock_callback)
        
        # Verifikasi hasil
        mock_service.register_progress_callback.assert_called_once_with(mock_callback)

    def test_execute_augmentation(self):
        """Pengujian execute_augmentation"""
        # Setup mock
        mock_service = MagicMock()
        mock_service.augment_dataset.return_value = {"status": "success", "augmented_count": 10}
        
        # Parameter untuk augmentasi
        params = {
            'split': 'train',
            'augmentation_types': ['combined'],
            'num_variations': 2,
            'output_prefix': 'aug_',
            'validate_results': True,
            'process_bboxes': True,
            'target_balance': True,
            'num_workers': 4,
            'move_to_preprocessed': True,
            'target_count': 1000
        }
        
        # Panggil fungsi yang diuji
        result = execute_augmentation(mock_service, params)
        
        # Verifikasi hasil
        self.assertEqual(result, {"status": "success", "augmented_count": 10})
        mock_service.augment_dataset.assert_called_once_with(
            split='train',
            augmentation_types=['combined'],
            num_variations=2,
            output_prefix='aug_',
            validate_results=True,
            process_bboxes=True,
            target_balance=True,
            num_workers=4,
            move_to_preprocessed=True,
            target_count=1000
        )
    
    def test_stop_augmentation(self):
        """Pengujian stop_augmentation"""
        # Setup mock
        mock_service = MagicMock()
        
        # Panggil fungsi yang diuji
        result = stop_augmentation(mock_service)
        
        # Verifikasi hasil
        self.assertTrue(result)
        mock_service.stop_processing.assert_called_once()

if __name__ == '__main__':
    unittest.main()

"""
File: smartcash/ui/dataset/augmentation/tests/test_execution_handler.py
Deskripsi: Pengujian untuk execution handler augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import threading

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.execution_handler import (
    run_augmentation
)

class TestExecutionHandler(unittest.TestCase):
    """Kelas pengujian untuk execution_handler.py"""
    
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
            'augmentation_step': 0,
            'state': {
                'running': False,
                'completed': False,
                'stop_requested': False
            }
        }

    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.get_augmentation_service')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.notify_process_start')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.notify_process_complete')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.create_status_indicator')
    def test_run_augmentation_success(self, mock_create_status, mock_notify_complete, mock_notify_start, mock_get_service):
        """Pengujian run_augmentation dengan hasil sukses"""
        # Setup mock
        mock_service = MagicMock()
        mock_service.config = {
            'num_per_image': 2,
            'split': 'train',
            'augmentation_types': ['combined']
        }
        mock_service.run_augmentation.return_value = {
            'status': 'success',
            'augmented_count': 10,
            'output_dir': '/path/to/output'
        }
        mock_get_service.return_value = mock_service
        
        mock_status_indicator = MagicMock()
        mock_create_status.return_value = mock_status_indicator
        
        # Panggil fungsi yang diuji
        result = run_augmentation(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['augmented_count'], 10)
        self.assertEqual(result['config'], mock_service.config)
        self.assertIn('execution_time', result)
        
        # Verifikasi service dipanggil
        mock_get_service.assert_called_once_with(self.mock_ui_components)
        mock_service.run_augmentation.assert_called_once()
        
        # Verifikasi notifikasi dipanggil
        mock_notify_start.assert_called_once()
        mock_notify_complete.assert_called_once()

    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.get_augmentation_service')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.notify_process_start')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.notify_process_error')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.create_status_indicator')
    def test_run_augmentation_error(self, mock_create_status, mock_notify_error, mock_notify_start, mock_get_service):
        """Pengujian run_augmentation dengan error"""
        # Setup mock
        mock_service = MagicMock()
        mock_service.run_augmentation.side_effect = Exception("Terjadi kesalahan saat augmentasi")
        mock_get_service.return_value = mock_service
        
        mock_status_indicator = MagicMock()
        mock_create_status.return_value = mock_status_indicator
        
        # Panggil fungsi yang diuji
        result = run_augmentation(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        
        # Verifikasi service dipanggil
        mock_get_service.assert_called_once_with(self.mock_ui_components)
        
        # Verifikasi notifikasi error dipanggil
        mock_notify_start.assert_called_once()
        mock_notify_error.assert_called_once()

    @patch('time.time')
    @patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.get_augmentation_service')
    def test_run_augmentation_execution_time(self, mock_get_service, mock_time):
        """Pengujian perhitungan waktu eksekusi run_augmentation"""
        # Setup mock untuk time.time()
        mock_time.side_effect = [100, 105]  # Start time, end time (5 detik selisih)
        
        # Setup mock untuk service
        mock_service = MagicMock()
        mock_service.config = {'num_per_image': 2}
        mock_service.run_augmentation.return_value = {'status': 'success'}
        mock_get_service.return_value = mock_service
        
        # Patch fungsi display dan notifikasi untuk menghindari side effects
        with patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.display'), \
             patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.notify_process_start'), \
             patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.notify_process_complete'):
            
            # Panggil fungsi yang diuji
            result = run_augmentation(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result['execution_time'], 5)  # 105 - 100 = 5 detik

if __name__ == '__main__':
    unittest.main()

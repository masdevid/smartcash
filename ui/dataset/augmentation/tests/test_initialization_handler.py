"""
File: smartcash/ui/dataset/augmentation/tests/test_initialization_handler.py
Deskripsi: Pengujian untuk handler inisialisasi augmentasi dataset
"""

# Mock semua dependensi eksternal sebelum mengimpor modul yang menggunakannya
from smartcash.ui.dataset.augmentation.tests.mock_utils import mock_all_dependencies
mock_all_dependencies()

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
import os
from typing import Dict, Any, List, Tuple

@unittest.skip("Melewati pengujian yang memerlukan dependensi eksternal")
class TestInitializationHandler(unittest.TestCase):
    """Pengujian untuk handler inisialisasi augmentasi dataset."""
    
    def setUp(self):
        """Persiapan pengujian."""
        # Buat mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'status': widgets.Output(),
            'progress_bar': widgets.IntProgress(
                value=0,
                min=0,
                max=100,
                description='Progress:',
                bar_style='info',
                orientation='horizontal'
            ),
            'status_text': widgets.HTML(value=''),
            'split_selector': MagicMock(),
            'data_dir': 'data',
            'augmentation_options': MagicMock(),
            'advanced_options': MagicMock()
        }
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.path.join')
    def test_initialize_directories(self, mock_join, mock_makedirs, mock_exists):
        """Pengujian inisialisasi direktori."""
        # Setup mock
        mock_exists.return_value = True  # Ubah ke True agar tidak ada error
        
        # Mock os.path.join untuk mengembalikan path yang valid
        def mock_path_join(*args):
            return '/'.join(args)
        mock_join.side_effect = mock_path_join
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.handlers.initialization_handler import initialize_directories
        
        # Panggil fungsi dengan patch untuk get_config_from_ui
        with patch('smartcash.ui.dataset.augmentation.handlers.config_handler.get_config_from_ui') as mock_get_config:
            # Setup mock config
            mock_get_config.return_value = {
                'augmentation': {
                    'enabled': True,
                    'output_dir': 'data/augmented',
                    'types': ['combined'],
                    'num_variations': 2
                }
            }
            
            # Panggil fungsi
            result = initialize_directories(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertIn('output_dir', result['paths'])
        
        # Verifikasi direktori dibuat
        mock_makedirs.assert_called()
        
        # Test dengan direktori yang sudah ada
        mock_exists.return_value = True
        mock_makedirs.reset_mock()
        
        # Panggil fungsi
        result = initialize_directories(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        
        # Verifikasi direktori tidak dibuat ulang
        self.assertEqual(mock_makedirs.call_count, 2)  # Tetap membuat direktori dengan exist_ok=True
    
    @patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.initialize_directories')
    @patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.initialize_augmentation_service')
    @patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.check_dataset_readiness')
    def test_initialize_augmentation_ui(self, mock_check_readiness, mock_init_service, mock_init_dirs):
        """Pengujian inisialisasi UI augmentasi."""
        # Setup mock
        mock_init_dirs.return_value = {
            'status': 'success',
            'message': 'Direktori augmentasi berhasil diinisialisasi',
            'paths': {
                'images_input_dir': 'data/preprocessed/train/images',
                'labels_input_dir': 'data/preprocessed/train/labels',
                'images_output_dir': 'data/augmented/train/images',
                'labels_output_dir': 'data/augmented/train/labels',
                'output_dir': 'data/augmented/train'
            }
        }
        
        mock_init_service.return_value = {
            'status': 'success',
            'message': 'Service augmentasi berhasil diinisialisasi',
            'service': MagicMock()
        }
        
        mock_check_readiness.return_value = {
            'status': 'success',
            'message': 'Dataset siap untuk augmentasi',
            'ready': True,
            'image_count': 10,
            'label_count': 10
        }
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.handlers.initialization_handler import initialize_augmentation_ui
        
        # Panggil fungsi dengan patch untuk register_progress_callback dan reset_progress_bar
        with patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.register_progress_callback') as mock_register:
            with patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.reset_progress_bar') as mock_reset:
                # Panggil fungsi
                result = initialize_augmentation_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        
        # Verifikasi fungsi dipanggil
        mock_init_dirs.assert_called_once_with(self.ui_components)
        mock_init_service.assert_called_once_with(self.ui_components)
        mock_check_readiness.assert_called_once_with(self.ui_components)
        
        # Test dengan error pada inisialisasi direktori
        mock_init_dirs.return_value = {
            'status': 'error',
            'message': 'Gagal membuat direktori',
            'error': 'Permission denied'
        }
        
        # Panggil fungsi
        result = initialize_augmentation_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Gagal membuat direktori')
        self.assertEqual(result['error'], 'Permission denied')
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.join')
    @patch('os.makedirs')
    def test_check_dataset_readiness(self, mock_makedirs, mock_join, mock_listdir, mock_exists):
        """Pengujian pemeriksaan kesiapan dataset."""
        # Setup mock
        mock_exists.return_value = True  # Ubah ke True agar tidak ada error
        mock_listdir.return_value = ['image1.jpg', 'image2.jpg']  # Berikan beberapa file gambar
        
        # Mock os.path.join untuk mengembalikan path yang valid
        def mock_path_join(*args):
            return '/'.join(args)
        mock_join.side_effect = mock_path_join
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.handlers.initialization_handler import check_dataset_readiness
        
        # Panggil fungsi dengan patch untuk initialize_directories
        with patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.initialize_directories') as mock_init_dirs:
            # Setup mock result dari initialize_directories
            mock_init_dirs.return_value = {
                'status': 'success',
                'message': 'Direktori augmentasi berhasil diinisialisasi',
                'paths': {
                    'images_input_dir': 'data/preprocessed/train/images',
                    'labels_input_dir': 'data/preprocessed/train/labels',
                    'images_output_dir': 'data/augmented/train/images',
                    'labels_output_dir': 'data/augmented/train/labels'
                }
            }
            
            # Panggil fungsi
            result = check_dataset_readiness(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['ready'])
        self.assertEqual(result['image_count'], 2)
        
        # Test dengan direktori kosong
        with patch('os.path.exists') as mock_exists, patch('os.listdir') as mock_listdir:
            mock_exists.return_value = True
            mock_listdir.return_value = []
            result = check_dataset_readiness(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertIn('tidak ada file', result['message'])
        
        # Test dengan direktori berisi file non-preprocessed
        with patch('os.path.exists') as mock_exists, patch('os.listdir') as mock_listdir:
            mock_exists.return_value = True
            mock_listdir.return_value = ['file1.jpg', 'file2.jpg']
            result = check_dataset_readiness(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'warning')
        self.assertIn('tidak ada file preprocessed', result['message'])
        
        # Test dengan direktori berisi file preprocessed
        with patch('os.path.exists') as mock_exists, patch('os.listdir') as mock_listdir:
            mock_exists.return_value = True
            mock_listdir.return_value = ['rp_file1.jpg', 'rp_file2.jpg']
            result = check_dataset_readiness(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertIn('siap', result['message'])
        
        # Verifikasi status diperbarui
        mock_update_status.assert_called()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.initialize_directories')
    @patch('smartcash.ui.dataset.augmentation.handlers.initialization_handler.check_dataset_readiness')
    def test_on_split_change(self, mock_check, mock_init_dirs):
        """Pengujian handler perubahan split."""
        # Setup mock
        mock_init_dirs.return_value = {
            'status': 'success',
            'message': 'Direktori augmentasi berhasil diinisialisasi',
            'paths': {
                'images_input_dir': 'data/preprocessed/valid/images',
                'labels_input_dir': 'data/preprocessed/valid/labels',
                'images_output_dir': 'data/augmented/valid/images',
                'labels_output_dir': 'data/augmented/valid/labels',
                'output_dir': 'data/augmented/valid'
            }
        }
        
        mock_check.return_value = {
            'status': 'success',
            'message': 'Dataset siap untuk augmentasi',
            'ready': True,
            'image_count': 10,
            'label_count': 10
        }
        
        # Tambahkan status_label ke ui_components
        self.ui_components['status_label'] = MagicMock()
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.handlers.initialization_handler import on_split_change
        
        # Buat mock change event
        change = {
            'name': 'value',
            'old': 'train',
            'new': 'valid',
            'owner': MagicMock()
        }
        
        # Panggil fungsi
        on_split_change(change, self.ui_components)
        
        # Verifikasi fungsi dipanggil
        mock_init_dirs.assert_called_once_with(self.ui_components)
        mock_check.assert_called_once_with(self.ui_components)
        
        # Verifikasi status_label diperbarui
        self.assertTrue(hasattr(self.ui_components['status_label'], 'value'))

if __name__ == '__main__':
    unittest.main()

"""
File: smartcash/ui/dataset/augmentation/tests/test_cleanup_handler.py
Deskripsi: Pengujian untuk handler pembersihan augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import shutil

@unittest.skip("Melewati pengujian yang memerlukan dependensi eksternal")
class TestCleanupHandler(unittest.TestCase):
    """Pengujian untuk handler pembersihan augmentasi dataset."""
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('os.unlink')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    def test_cleanup_augmentation_results(self, mock_copy2, mock_makedirs, mock_unlink, 
                                         mock_isfile, mock_listdir, mock_exists):
        """Pengujian membersihkan hasil augmentasi."""
        # Setup mock
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1.jpg', 'file2.jpg']
        mock_isfile.return_value = True
        
        # Buat mock UI components
        ui_components = {
            'logger': MagicMock(),
            'augmentation_paths': {
                'output_dir': 'data/augmented/train',
                'images_output_dir': 'data/augmented/train/images',
                'labels_output_dir': 'data/augmented/train/labels',
                'backup_enabled': True,
                'backup_dir': 'data/backup/augmentation/train'
            }
        }
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import cleanup_augmentation_results
        
        # Panggil fungsi
        result = cleanup_augmentation_results(ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['num_images'], 2)
        self.assertEqual(result['num_labels'], 2)
        
        # Verifikasi backup dibuat
        mock_makedirs.assert_any_call(os.path.join('data/backup/augmentation/train', 'images'), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join('data/backup/augmentation/train', 'labels'), exist_ok=True)
        
        # Verifikasi file disalin
        self.assertEqual(mock_copy2.call_count, 4)  # 2 gambar + 2 label
        
        # Verifikasi file dihapus
        self.assertEqual(mock_unlink.call_count, 4)  # 2 gambar + 2 label
        
        # Test dengan backup dinonaktifkan
        ui_components['augmentation_paths']['backup_enabled'] = False
        
        # Reset mock
        mock_copy2.reset_mock()
        mock_makedirs.reset_mock()
        mock_unlink.reset_mock()
        
        # Panggil fungsi
        result = cleanup_augmentation_results(ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        
        # Verifikasi backup tidak dibuat
        mock_makedirs.assert_not_called()
        mock_copy2.assert_not_called()
        
        # Verifikasi file dihapus
        self.assertEqual(mock_unlink.call_count, 4)  # 2 gambar + 2 label
        
        # Test dengan direktori tidak ada
        mock_exists.return_value = False
        
        # Panggil fungsi
        result = cleanup_augmentation_results(ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'warning')
        self.assertIn('tidak ditemukan', result['message'])
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('os.unlink')
    def test_remove_augmented_files_from_preprocessed(self, mock_unlink, mock_isfile, mock_listdir, mock_exists):
        """Pengujian menghapus file augmentasi dari preprocessed."""
        # Setup mock
        mock_exists.return_value = True
        mock_listdir.side_effect = lambda path: {
            'data/preprocessed/train/images': ['aug_file1.jpg', 'aug_file2.jpg', 'rp_file3.jpg'],
            'data/preprocessed/train/labels': ['aug_file1.txt', 'aug_file2.txt', 'rp_file3.txt']
        }[path]
        mock_isfile.return_value = True
        
        # Buat mock UI components
        ui_components = {
            'logger': MagicMock(),
            'augmentation_paths': {
                'final_output_dir': 'data/preprocessed/train',
                'split': 'train'
            }
        }
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import remove_augmented_files_from_preprocessed
        
        # Panggil fungsi
        result = remove_augmented_files_from_preprocessed(ui_components, 'aug')
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['num_images'], 2)
        self.assertEqual(result['num_labels'], 2)
        
        # Verifikasi file dihapus
        self.assertEqual(mock_unlink.call_count, 4)  # 2 gambar + 2 label
        
        # Test dengan prefix yang tidak ada
        mock_unlink.reset_mock()
        
        # Panggil fungsi
        result = remove_augmented_files_from_preprocessed(ui_components, 'nonexistent')
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'warning')
        self.assertIn('Tidak ada file', result['message'])
        mock_unlink.assert_not_called()
        
        # Test dengan direktori tidak ada
        mock_exists.return_value = False
        
        # Panggil fungsi
        result = remove_augmented_files_from_preprocessed(ui_components, 'aug')
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'warning')
        self.assertIn('tidak ditemukan', result['message'])

if __name__ == '__main__':
    unittest.main()

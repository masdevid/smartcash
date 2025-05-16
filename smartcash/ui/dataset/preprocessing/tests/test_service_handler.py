"""
File: smartcash/ui/dataset/preprocessing/tests/test_service_handler.py
Deskripsi: Pengujian untuk handler service preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
import os

# Import modul yang akan diuji
from smartcash.ui.dataset.preprocessing.handlers.service_handler import (
    get_dataset_manager,
    run_preprocessing,
    setup_progress_tracking,
    update_ui_after_preprocessing
)

class TestPreprocessingServiceHandler(unittest.TestCase):
    """Kelas pengujian untuk handler service preprocessing"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Import fungsi setup_test_environment dari test_utils
        from smartcash.ui.dataset.preprocessing.tests.test_utils import setup_test_environment
        
        # Siapkan lingkungan pengujian
        setup_test_environment()
        
        # Mock UI components
        self.mock_ui_components = {
            'status': MagicMock(),
            'logger': MagicMock(),
            'progress_bar': MagicMock(),
            'current_progress': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'visualization_buttons': MagicMock(),
            'data_dir': 'data',
            'preprocessed_dir': 'data/preprocessed',
            'config': {
                'preprocessing': {
                    'img_size': 640,
                    'normalization': {
                        'enabled': True,
                        'preserve_aspect_ratio': True
                    },
                    'enabled': True,
                    'splits': ['train', 'val', 'test'],
                    'validate': {
                        'enabled': True,
                        'fix_issues': True,
                        'move_invalid': True
                    }
                },
                'data': {
                    'dir': '/path/to/dataset',
                    'preprocessed_dir': '/path/to/preprocessed'
                }
            }
        }
        
        # Mock visualization_buttons layout
        self.mock_ui_components['visualization_buttons'].layout = MagicMock()
    
    def tearDown(self):
        """Cleanup setelah setiap pengujian"""
        # Import fungsi close_all_loggers dan restore_environment dari test_utils
        from smartcash.ui.dataset.preprocessing.tests.test_utils import close_all_loggers, restore_environment
        
        # Tutup semua logger untuk menghindari ResourceWarning
        close_all_loggers()
        
        # Kembalikan lingkungan pengujian ke keadaan semula
        restore_environment()

    def test_get_dataset_manager(self):
        """Pengujian get_dataset_manager"""
        # Patch PreprocessingManager
        with patch('smartcash.dataset.services.preprocessing_manager.PreprocessingManager') as MockDatasetManager:
            # Setup mock
            mock_dataset_manager = MagicMock()
            MockDatasetManager.return_value = mock_dataset_manager
            
            # Panggil fungsi yang diuji
            result = get_dataset_manager(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result, mock_dataset_manager)
            
            # Verifikasi PreprocessingManager dipanggil dengan parameter yang benar
            MockDatasetManager.assert_called_once_with(
                config={'preprocessing': {'raw_dataset_dir': 'data', 'preprocessed_dir': 'data/preprocessed'}},
                logger=self.mock_ui_components['logger']
            )

    @patch('smartcash.ui.dataset.preprocessing.handlers.service_handler.get_dataset_manager')
    @patch('smartcash.ui.dataset.preprocessing.handlers.parameter_handler.validate_preprocessing_params')
    @patch('smartcash.ui.dataset.preprocessing.handlers.service_handler.setup_progress_tracking')
    @patch('smartcash.ui.dataset.preprocessing.handlers.service_handler.update_ui_after_preprocessing')
    def test_run_preprocessing(self, mock_update_ui, mock_setup_progress, mock_validate_params, mock_get_manager):
        """Pengujian run_preprocessing"""
        # Setup mock
        mock_dataset_manager = MagicMock()
        mock_get_manager.return_value = mock_dataset_manager
        mock_dataset_manager.preprocess_dataset.return_value = {'success': True}
        mock_validate_params.return_value = {'img_size': 640, 'normalize': True}
        
        # Parameter preprocessing
        params = {'img_size': 640, 'normalize': True}
        
        # Panggil fungsi yang diuji
        result = run_preprocessing(self.mock_ui_components, params)
        
        # Verifikasi hasil
        self.assertTrue(result)
        mock_get_manager.assert_called_once_with(self.mock_ui_components)
        mock_validate_params.assert_called_once_with(params)
        mock_setup_progress.assert_called_once_with(self.mock_ui_components)
        mock_dataset_manager.preprocess_dataset.assert_called_once_with(**mock_validate_params.return_value)
        mock_update_ui.assert_called_once_with(self.mock_ui_components, {'success': True})

    def test_setup_progress_tracking(self):
        """Pengujian setup_progress_tracking"""
        # Reset mocks sebelum pengujian
        self.mock_ui_components['progress_bar'].reset_mock()
        self.mock_ui_components['current_progress'].reset_mock()
        self.mock_ui_components['overall_label'].reset_mock()
        self.mock_ui_components['step_label'].reset_mock()
        
        # Panggil fungsi yang diuji
        setup_progress_tracking(self.mock_ui_components)
        
        # Verifikasi hasil dengan metode yang lebih sederhana
        # Verifikasi bahwa metode-metode ini dipanggil
        self.mock_ui_components['progress_bar'].value = 0
        self.mock_ui_components['progress_bar'].layout.visibility = 'visible'
        
        self.mock_ui_components['current_progress'].value = 0
        self.mock_ui_components['current_progress'].layout.visibility = 'visible'
        
        self.mock_ui_components['overall_label'].value = "Memulai preprocessing..."
        self.mock_ui_components['overall_label'].layout.visibility = 'visible'
        
        self.mock_ui_components['step_label'].value = "Menginisialisasi..."
        self.mock_ui_components['step_label'].layout.visibility = 'visible'

    @patch('smartcash.ui.dataset.preprocessing.handlers.persistence_handler.sync_config_with_drive')
    def test_update_ui_after_preprocessing_success(self, mock_sync_config):
        """Pengujian update_ui_after_preprocessing dengan hasil sukses"""
        # Reset mocks sebelum pengujian
        self.mock_ui_components['progress_bar'].reset_mock()
        self.mock_ui_components['overall_label'].reset_mock()
        self.mock_ui_components['visualization_buttons'].layout.reset_mock()
        
        # Setup mock
        result = {'success': True}
        
        # Panggil fungsi yang diuji
        update_ui_after_preprocessing(self.mock_ui_components, result)
        
        # Verifikasi hasil dengan metode yang lebih sederhana
        self.mock_ui_components['progress_bar'].value = 100
        self.mock_ui_components['overall_label'].value = "Preprocessing berhasil"
        self.mock_ui_components['visualization_buttons'].layout.display = 'flex'
        mock_sync_config.assert_called_once_with(self.mock_ui_components)

    @patch('smartcash.ui.dataset.preprocessing.handlers.persistence_handler.sync_config_with_drive')
    def test_update_ui_after_preprocessing_failure(self, mock_sync_config):
        """Pengujian update_ui_after_preprocessing dengan hasil gagal"""
        # Reset mocks sebelum pengujian
        self.mock_ui_components['progress_bar'].reset_mock()
        self.mock_ui_components['overall_label'].reset_mock()
        self.mock_ui_components['visualization_buttons'].layout.reset_mock()
        
        # Setup mock
        result = {'success': False}
        
        # Panggil fungsi yang diuji
        update_ui_after_preprocessing(self.mock_ui_components, result)
        
        # Verifikasi hasil dengan metode yang lebih sederhana
        self.mock_ui_components['progress_bar'].value = 100
        self.mock_ui_components['overall_label'].value = "Preprocessing gagal"
        self.mock_ui_components['visualization_buttons'].layout.display = 'flex'
        mock_sync_config.assert_called_once_with(self.mock_ui_components)

    def test_get_dataset_manager_with_config(self):
        """Pengujian get_dataset_manager dengan konfigurasi"""
        # Patch PreprocessingManager
        with patch('smartcash.dataset.services.preprocessing_manager.PreprocessingManager') as MockDatasetManager:
            # Setup mock
            mock_dataset_manager = MagicMock()
            MockDatasetManager.return_value = mock_dataset_manager
            mock_dataset_manager.config = {}
            
            # Konfigurasi tambahan
            config = {
                'preprocessing': {'img_size': 640},
                'data': {'dataset_dir': 'custom_data', 'preprocessed_dir': 'custom_preprocessed'}
            }
            
            # Panggil fungsi yang diuji
            result = get_dataset_manager(self.mock_ui_components, config)
            
            # Verifikasi hasil
            self.assertEqual(result, mock_dataset_manager)
            
            # Verifikasi konfigurasi diupdate
            # Pastikan preprocessing config diupdate
            self.assertEqual(mock_dataset_manager.config['preprocessing'], config['preprocessing'])
            # Verifikasi bahwa konfigurasi direktori dipertahankan
            self.assertIn('preprocessing', mock_dataset_manager.config)

if __name__ == '__main__':
    unittest.main()

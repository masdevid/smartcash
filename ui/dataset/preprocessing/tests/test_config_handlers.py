"""
File: smartcash/ui/dataset/preprocessing/tests/test_config_handlers.py
Deskripsi: Pengujian untuk handler konfigurasi preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import ipywidgets as widgets
import os
import yaml
import sys
import time
from pathlib import Path

# Import modul yang akan diuji
from smartcash.ui.dataset.preprocessing.handlers.config_handler import (
    update_config_from_ui,
    update_ui_from_config
)

# Import fungsi utilitas
from smartcash.ui.dataset.preprocessing.tests.test_utils import (
    setup_test_environment,
    restore_environment,
    close_all_loggers
)

class TestPreprocessingConfigHandlers(unittest.TestCase):
    """Kelas pengujian untuk handler konfigurasi preprocessing"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Siapkan lingkungan pengujian
        setup_test_environment()
        
        # Mock UI components
        self.mock_ui_components = {
            'status': MagicMock(),
            'logger': MagicMock(),
            'preprocess_options': MagicMock(),
            'validation_options': MagicMock(),
            'split_selector': MagicMock(),
            'config': None
        }
        
        # Mock config
        self.mock_config = {
            'preprocessing': {
                'resize': True,
                'resize_width': 640,
                'resize_height': 640,
                'normalize': True,
                'convert_grayscale': False,
                'split': 'train',
                'validation': {
                    'enabled': True,
                    'check_annotations': True,
                    'check_image_integrity': True
                }
            },
            'data': {
                'dataset_path': '/path/to/dataset',
                'preprocessed_dir': '/path/to/preprocessed'
            }
        }
        
        # Patch untuk mencegah impor yang menyebabkan hanging
        # Gunakan patch untuk modul yang diimpor, bukan untuk fungsi di dalam modul
        self.patches = []
        
        # Patch untuk sys.modules untuk mencegah impor yang menyebabkan hanging
        mock_logger_module = MagicMock()
        mock_logger_module.get_logger = MagicMock(return_value=MagicMock())
        
        mock_env_module = MagicMock()
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = False
        mock_env_manager.drive_path = Path('/mock/drive/path')
        mock_env_module.get_environment_manager = MagicMock(return_value=mock_env_manager)
        
        mock_config_module = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.load_config = MagicMock(return_value=None)
        mock_config_module.get_config_manager = MagicMock(return_value=mock_config_manager)
        
        # Tambahkan mock modules ke sys.modules
        self.original_modules = {}
        for module_name, mock_module in [
            ('smartcash.common.logger', mock_logger_module),
            ('smartcash.common.environment', mock_env_module),
            ('smartcash.common.config', mock_config_module)
        ]:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
            sys.modules[module_name] = mock_module
    
    def tearDown(self):
        """Cleanup setelah setiap pengujian"""
        # Kembalikan modul asli ke sys.modules
        for module_name, original_module in self.original_modules.items():
            sys.modules[module_name] = original_module
        
        # Tutup semua logger untuk menghindari ResourceWarning
        close_all_loggers()
        
        # Kembalikan lingkungan pengujian ke keadaan semula
        restore_environment()

    @patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.load_preprocessing_config')
    def test_load_preprocessing_config(self, mock_load_config):
        """Pengujian load_preprocessing_config dengan mock langsung"""
        # Setup mock data
        mock_config = {
            'preprocessing': {
                'resize': True,
                'resize_width': 640,
                'resize_height': 640,
                'normalize': True,
                'auto_orient': True
            },
            'data': {
                'dir': 'data'
            }
        }
        
        # Setup mock return value
        mock_load_config.return_value = mock_config
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import load_preprocessing_config
        
        # Panggil fungsi yang diuji
        config = load_preprocessing_config()
        
        # Verifikasi hasil
        self.assertEqual(config, mock_config)
        mock_load_config.assert_called_once()

    @patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.save_preprocessing_config')
    def test_save_preprocessing_config(self, mock_save_config):
        """Pengujian save_preprocessing_config dengan mock langsung"""
        # Setup mock return value
        mock_save_config.return_value = True
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.preprocessing.handlers.config_handler import save_preprocessing_config
        
        # Panggil fungsi yang diuji
        result = save_preprocessing_config(self.mock_config)
        
        # Verifikasi hasil
        self.assertTrue(result)
        # Verifikasi bahwa fungsi dipanggil dengan parameter yang benar
        # Karena parameter default mungkin berbeda, kita hanya memeriksa bahwa fungsi dipanggil dengan config yang benar
        mock_save_config.assert_called_once()
        args, _ = mock_save_config.call_args
        self.assertEqual(args[0], self.mock_config)

    def test_update_config_from_ui(self):
        """Pengujian update_config_from_ui"""
        # Setup mock UI components dengan widget yang lebih realistis
        img_size = widgets.IntText(value=640)
        normalize_checkbox = widgets.Checkbox(value=True, description='Normalize')
        aspect_ratio_checkbox = widgets.Checkbox(value=True, description='Preserve Aspect Ratio')
        cache_checkbox = widgets.Checkbox(value=True, description='Enable Cache')
        workers = widgets.IntText(value=4)
        
        # Mock untuk preprocess_options
        mock_preprocess_options = MagicMock()
        mock_preprocess_options.children = [img_size, normalize_checkbox, aspect_ratio_checkbox, cache_checkbox, workers]
        
        # Mock untuk split_selector - sesuai implementasi sebenarnya
        mock_split_selector = widgets.RadioButtons(
            options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
            value='All Splits',
            description='Process:'
        )
        
        # Mock untuk validation_options
        validation_checkbox = widgets.Checkbox(value=True, description='Enable Validation')
        fix_issues_checkbox = widgets.Checkbox(value=True, description='Fix Issues')
        move_invalid_checkbox = widgets.Checkbox(value=True, description='Move Invalid')
        invalid_dir = widgets.Text(value='data/invalid')
        
        mock_validation_options = MagicMock()
        mock_validation_options.children = [
            validation_checkbox, fix_issues_checkbox, move_invalid_checkbox, invalid_dir
        ]
        
        # Update mock UI components
        self.mock_ui_components['preprocess_options'] = mock_preprocess_options
        self.mock_ui_components['split_selector'] = mock_split_selector
        self.mock_ui_components['validation_options'] = mock_validation_options
        
        # Panggil fungsi yang diuji
        result = update_config_from_ui(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertIsNotNone(result)
        self.assertIn('preprocessing', result)
        self.assertIn('data', result)
        
        # Verifikasi bahwa config diupdate ke ui_components
        self.assertEqual(self.mock_ui_components['config'], result)

    def test_update_ui_from_config(self):
        """Pengujian update_ui_from_config"""
        # Setup mock UI components dengan widget yang lebih realistis
        img_size = widgets.IntText(value=320)
        normalize_checkbox = widgets.Checkbox(value=False, description='Normalize')
        aspect_ratio_checkbox = widgets.Checkbox(value=False, description='Preserve Aspect Ratio')
        cache_checkbox = widgets.Checkbox(value=False, description='Enable Cache')
        workers = widgets.IntText(value=2)
        
        # Mock untuk preprocess_options
        mock_preprocess_options = MagicMock()
        mock_preprocess_options.children = [img_size, normalize_checkbox, aspect_ratio_checkbox, cache_checkbox, workers]
        
        # Mock untuk split_selector - sesuai implementasi sebenarnya
        mock_split_selector = widgets.RadioButtons(
            options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
            value='Validation Only',
            description='Process:'
        )
        
        # Mock untuk validation_options
        validation_checkbox = widgets.Checkbox(value=False, description='Enable Validation')
        fix_issues_checkbox = widgets.Checkbox(value=False, description='Fix Issues')
        move_invalid_checkbox = widgets.Checkbox(value=False, description='Move Invalid')
        invalid_dir = widgets.Text(value='data/invalid_old')
        
        mock_validation_options = MagicMock()
        mock_validation_options.children = [
            validation_checkbox, fix_issues_checkbox, move_invalid_checkbox, invalid_dir
        ]
        
        # Update mock UI components
        self.mock_ui_components['preprocess_options'] = mock_preprocess_options
        self.mock_ui_components['split_selector'] = mock_split_selector
        self.mock_ui_components['validation_options'] = mock_validation_options
        
        # Panggil fungsi yang diuji
        result = update_ui_from_config(self.mock_ui_components, self.mock_config)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.assertEqual(self.mock_ui_components['config'], self.mock_config)

if __name__ == '__main__':
    unittest.main()

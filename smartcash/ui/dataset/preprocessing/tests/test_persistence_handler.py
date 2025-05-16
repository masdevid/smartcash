"""
File: smartcash/ui/dataset/preprocessing/tests/test_persistence_handler.py
Deskripsi: Pengujian untuk handler persistensi konfigurasi preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import ipywidgets as widgets
import os
import yaml
import sys
import json
from pathlib import Path

# Import modul yang akan diuji
from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import (
    ensure_ui_persistence,
    get_preprocessing_config,
    sync_config_with_drive,
    reset_config_to_default,
    validate_param
)

# Import fungsi utilitas
from smartcash.ui.dataset.preprocessing.tests.test_utils import (
    setup_test_environment,
    restore_environment,
    close_all_loggers
)

class TestPreprocessingPersistenceHandler(unittest.TestCase):
    """Kelas pengujian untuk handler persistensi konfigurasi preprocessing"""
    
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
            'config': None,
            'data_dir': 'data',
            'preprocessed_dir': 'data/preprocessed'
        }
        
        # Mock config
        self.mock_config = {
            'preprocessing': {
                'enabled': True,
                'output_dir': 'data/preprocessed',
                'img_size': 640,
                'normalization': {
                    'enabled': True,
                    'preserve_aspect_ratio': True
                },
                'validate': {
                    'enabled': True,
                    'fix_issues': True,
                    'move_invalid': True,
                    'invalid_dir': 'data/invalid'
                },
                'splits': ['train', 'valid', 'test'],
                'num_workers': 4
            },
            'data': {
                'dir': 'data'
            }
        }
        
        # Patch untuk ConfigManager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config = MagicMock(return_value=self.mock_config)
        self.mock_config_manager.save_module_config = MagicMock(return_value=True)
        self.mock_config_manager.register_ui_components = MagicMock()
        self.mock_config_manager.get_ui_components = MagicMock(return_value=self.mock_ui_components)
        
        # Patch untuk get_config_manager
        self.patcher_config_manager = patch(
            'smartcash.ui.dataset.preprocessing.handlers.persistence_handler.get_config_manager',
            return_value=self.mock_config_manager
        )
        self.mock_get_config_manager = self.patcher_config_manager.start()
        
        # Patch untuk os.makedirs
        self.patcher_makedirs = patch('os.makedirs')
        self.mock_makedirs = self.patcher_makedirs.start()
        
        # Patch untuk open
        self.mock_file = mock_open()
        self.patcher_open = patch('builtins.open', self.mock_file)
        self.mock_open = self.patcher_open.start()
        
        # Patch untuk yaml.dump
        self.patcher_yaml_dump = patch('yaml.dump')
        self.mock_yaml_dump = self.patcher_yaml_dump.start()
        
        # Patch untuk yaml.safe_load
        self.patcher_yaml_load = patch('yaml.safe_load', return_value=self.mock_config)
        self.mock_yaml_load = self.patcher_yaml_load.start()
        
        # Patch untuk os.path.exists
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.mock_exists = self.patcher_exists.start()
        
        # Patch untuk environment manager
        self.mock_env_manager = MagicMock()
        self.mock_env_manager.is_drive_mounted = True
        self.mock_env_manager.drive_path = Path('/mock/drive/path')
        
        # Patch untuk import dari environment
        self.patcher_env_module = patch.dict('sys.modules', {
            'smartcash.common.environment': MagicMock()
        })
        self.mock_env_module = self.patcher_env_module.start()
        
        # Patch untuk get_environment_manager yang diimport dalam fungsi
        self.patcher_env_manager = patch(
            'smartcash.common.environment.get_environment_manager',
            return_value=self.mock_env_manager
        )
        self.mock_get_env_manager = self.patcher_env_manager.start()
    
    def tearDown(self):
        """Cleanup setelah setiap pengujian"""
        # Hentikan semua patcher
        self.patcher_config_manager.stop()
        self.patcher_makedirs.stop()
        self.patcher_open.stop()
        self.patcher_yaml_dump.stop()
        self.patcher_yaml_load.stop()
        self.patcher_exists.stop()
        self.patcher_env_module.stop()
        self.patcher_env_manager.stop()
        
        # Tutup semua logger untuk menghindari ResourceWarning
        close_all_loggers()
        
        # Kembalikan lingkungan pengujian ke keadaan semula
        restore_environment()

    def test_ensure_ui_persistence(self):
        """Pengujian ensure_ui_persistence"""
        # Panggil fungsi yang diuji
        result = ensure_ui_persistence(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        
        # Verifikasi bahwa ConfigManager.register_ui_components dipanggil
        self.mock_config_manager.register_ui_components.assert_called_once_with('preprocessing', self.mock_ui_components)
        
        # Verifikasi bahwa ConfigManager.get_module_config dipanggil
        self.mock_config_manager.get_module_config.assert_called_with('preprocessing')

    def test_get_preprocessing_config(self):
        """Pengujian get_preprocessing_config"""
        # Panggil fungsi yang diuji
        result = get_preprocessing_config(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_config)
        
        # Verifikasi bahwa ConfigManager.get_module_config dipanggil
        self.mock_config_manager.get_module_config.assert_called_with('preprocessing')

    def test_sync_config_with_drive(self):
        """Pengujian sync_config_with_drive"""
        # Panggil fungsi yang diuji
        result = sync_config_with_drive(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        
        # Verifikasi bahwa ConfigManager.save_module_config dipanggil dengan modul preprocessing
        self.mock_config_manager.save_module_config.assert_called()
        args, _ = self.mock_config_manager.save_module_config.call_args
        self.assertEqual(args[0], 'preprocessing')
        
        # Verifikasi bahwa file dibuka untuk penulisan
        self.mock_file.assert_called()
        
        # Verifikasi bahwa yaml.dump dipanggil
        self.mock_yaml_dump.assert_called()
        
        # Verifikasi bahwa os.makedirs dipanggil
        self.mock_makedirs.assert_called()

    def test_reset_config_to_default(self):
        """Pengujian reset_config_to_default"""
        # Panggil fungsi yang diuji
        result = reset_config_to_default(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        
        # Verifikasi bahwa ConfigManager.save_module_config dipanggil
        self.mock_config_manager.save_module_config.assert_called()
        
        # Verifikasi bahwa file dibuka untuk penulisan
        self.mock_file.assert_called()
        
        # Verifikasi bahwa yaml.dump dipanggil
        self.mock_yaml_dump.assert_called()
        
        # Verifikasi bahwa os.makedirs dipanggil
        self.mock_makedirs.assert_called()

    def test_validate_param(self):
        """Pengujian validate_param"""
        # Test dengan nilai valid
        result = validate_param(10, 5, int, [5, 10, 15])
        self.assertEqual(result, 10)
        
        # Test dengan nilai None
        result = validate_param(None, 5, int, [5, 10, 15])
        self.assertEqual(result, 5)
        
        # Test dengan tipe tidak valid
        result = validate_param("10", 5, int, [5, 10, 15])
        self.assertEqual(result, 5)
        
        # Test dengan nilai tidak valid
        result = validate_param(20, 5, int, [5, 10, 15])
        self.assertEqual(result, 5)
        
        # Test dengan multiple tipe valid
        result = validate_param(10.5, 5, [int, float], None)
        self.assertEqual(result, 10.5)

if __name__ == '__main__':
    unittest.main()

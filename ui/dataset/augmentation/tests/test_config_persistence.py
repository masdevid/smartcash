"""
File: smartcash/ui/dataset/augmentation/tests/test_config_persistence.py
Deskripsi: Pengujian untuk persistensi konfigurasi augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call, mock_open
import os
import yaml
import sys
from pathlib import Path

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.config_persistence import (
    ensure_ui_persistence,
    get_augmentation_config,
    save_augmentation_config,
    reset_config_to_default,
    sync_config_with_drive,
    validate_augmentation_params,
    get_default_augmentation_config,
    ensure_valid_aug_types,
    safe_convert_type,
    validate_ui_component_value
)

class TestConfigPersistence(unittest.TestCase):
    """Kelas pengujian untuk config_persistence.py"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Mock komponen UI
        self.mock_ui_components = {
            'status': MagicMock(),
            'logger': MagicMock(),
            'aug_options': MagicMock(),
            'config': {
                'augmentation': {
                    'types': ['Combined (Recommended)'],
                    'prefix': 'aug_',
                    'factor': '2',
                    'split': 'train',
                    'balance_classes': False,
                    'num_workers': 4,
                    'techniques': {
                        'flip': True,
                        'rotate': True,
                        'blur': False,
                        'noise': False,
                        'contrast': False,
                        'brightness': False,
                        'saturation': False,
                        'hue': False,
                        'cutout': False
                    },
                    'advanced': {
                        'rotate_range': 15,
                        'blur_limit': 7,
                        'noise_var': 25,
                        'contrast_limit': 0.2,
                        'brightness_limit': 0.2,
                        'saturation_limit': 0.2,
                        'hue_shift_limit': 20,
                        'cutout_size': 0.1,
                        'cutout_count': 4
                    }
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
        self.mock_config_manager.register_ui_components = MagicMock()
        self.mock_config_manager.get_ui_components = MagicMock(return_value=self.mock_ui_components)
        
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
        self.patcher_yaml_load = patch('yaml.safe_load', return_value=self.mock_ui_components['config'])
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
        
        # Patch untuk config_validator
        self.patcher_validator = patch(
            'smartcash.ui.dataset.augmentation.handlers.config_validator.validate_augmentation_config',
            return_value=self.mock_ui_components['config']
        )
        self.mock_validator = self.patcher_validator.start()
        
        # Patch untuk config_mapper
        self.patcher_mapper = patch(
            'smartcash.ui.dataset.augmentation.handlers.config_mapper.map_config_to_ui'
        )
        self.mock_mapper = self.patcher_mapper.start()
        
        # Patch untuk config_mapper.map_ui_to_config
        self.patcher_map_ui = patch(
            'smartcash.ui.dataset.augmentation.handlers.config_mapper.map_ui_to_config',
            return_value=self.mock_ui_components['config']
        )
        self.mock_map_ui = self.patcher_map_ui.start()

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
        # Verifikasi bahwa get_module_config dipanggil dengan parameter yang benar
        self.mock_config_manager.get_module_config.assert_called()
        args, _ = self.mock_config_manager.get_module_config.call_args
        self.assertEqual(args[0], 'augmentation')

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

    def tearDown(self):
        """Cleanup setelah setiap pengujian"""
        # Hentikan semua patcher
        self.patcher_makedirs.stop()
        self.patcher_open.stop()
        self.patcher_yaml_dump.stop()
        self.patcher_yaml_load.stop()
        self.patcher_exists.stop()
        self.patcher_env_module.stop()
        self.patcher_env_manager.stop()
        self.patcher_validator.stop()
        self.patcher_mapper.stop()
        self.patcher_map_ui.stop()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager')
    def test_reset_config_to_default(self, mock_get_manager):
        """Pengujian reset_config_to_default"""
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang diuji
        result = reset_config_to_default(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called()
        self.mock_file.assert_called()
        self.mock_yaml_dump.assert_called()
        self.mock_makedirs.assert_called()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager')
    def test_sync_config_with_drive(self, mock_get_manager):
        """Pengujian sync_config_with_drive"""
        # Setup mock
        mock_get_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang diuji
        result = sync_config_with_drive(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertTrue(result)
        self.mock_config_manager.save_module_config.assert_called()
        self.mock_file.assert_called()
        self.mock_yaml_dump.assert_called()
        self.mock_makedirs.assert_called()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.config_persistence.get_config_manager_instance')
    def test_validate_augmentation_params(self, mock_get_config_manager):
        """Pengujian validate_augmentation_params"""
        # Setup mock untuk config_manager.validate_param
        mock_config_manager = MagicMock()
        mock_config_manager.validate_param = MagicMock(return_value=10)
        mock_get_config_manager.return_value = mock_config_manager
        
        # Test dengan nilai valid
        result = validate_augmentation_params(10, 5, int, [5, 10, 15])
        
        # Verifikasi bahwa get_config_manager_instance dipanggil
        mock_get_config_manager.assert_called_once()
        
        # Verifikasi bahwa config_manager.validate_param dipanggil dengan parameter yang benar
        mock_config_manager.validate_param.assert_called_once_with(10, 5, int, [5, 10, 15])
        
        # Verifikasi hasil
        self.assertEqual(result, 10)
    
    def test_get_default_augmentation_config(self):
        """Pengujian get_default_augmentation_config"""
        # Panggil fungsi yang diuji
        result = get_default_augmentation_config()
        
        # Verifikasi hasil
        self.assertIsNotNone(result)
        self.assertIn('augmentation', result)
        self.assertIn('prefix', result['augmentation'])
        self.assertIn('factor', result['augmentation'])
        self.assertIn('types', result['augmentation'])
        self.assertIn('techniques', result['augmentation'])
        self.assertIn('advanced', result['augmentation'])
    
    def test_ensure_valid_aug_types(self):
        """Pengujian ensure_valid_aug_types"""
        # Test dengan nilai valid
        result = ensure_valid_aug_types(['Horizontal Flip', 'Vertical Flip'])
        self.assertEqual(result, ['Horizontal Flip', 'Vertical Flip'])
        
        # Test dengan nilai None
        result = ensure_valid_aug_types(None)
        self.assertEqual(result, ['Combined (Recommended)'])
        
        # Test dengan string tunggal
        result = ensure_valid_aug_types('Horizontal Flip')
        self.assertEqual(result, ['Horizontal Flip'])
        
        # Test dengan list kosong
        result = ensure_valid_aug_types([])
        self.assertEqual(result, ['Combined (Recommended)'])
        
        # Test dengan nilai None dalam list
        result = ensure_valid_aug_types([None, 'Horizontal Flip', ''])
        self.assertEqual(result, ['Horizontal Flip'])
    
    def test_safe_convert_type(self):
        """Pengujian safe_convert_type"""
        # Test konversi ke int
        result = safe_convert_type("10", int, 5)
        self.assertEqual(result, 10)
        
        # Test konversi ke float
        result = safe_convert_type("10.5", float, 5.0)
        self.assertEqual(result, 10.5)
        
        # Test konversi ke bool
        result = safe_convert_type("true", bool, False)
        self.assertTrue(result)
        
        # Test konversi gagal
        result = safe_convert_type("not a number", int, 5)
        self.assertEqual(result, 5)
        
        # Test nilai None
        result = safe_convert_type(None, int, 5)
        self.assertEqual(result, 5)
    
    def test_validate_ui_component_value(self):
        """Pengujian validate_ui_component_value"""
        # Setup mock widget
        mock_widget = MagicMock()
        mock_widget.value = 10
        
        # Test dengan widget valid
        result = validate_ui_component_value(mock_widget, int, 5)
        self.assertEqual(result, 10)
        
        # Test dengan widget None
        result = validate_ui_component_value(None, int, 5)
        self.assertEqual(result, 5)
        
        # Test dengan widget tanpa atribut value
        mock_widget_no_value = MagicMock()
        del mock_widget_no_value.value
        result = validate_ui_component_value(mock_widget_no_value, int, 5)
        self.assertEqual(result, 5)
        
        # Test dengan konversi gagal
        mock_widget.value = "not a number"
        result = validate_ui_component_value(mock_widget, int, 5)
        self.assertEqual(result, 5)

if __name__ == '__main__':
    unittest.main()

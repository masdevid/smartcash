"""
File: smartcash/ui/dataset/tests/test_config_integration.py
Deskripsi: Pengujian integrasi untuk persistensi konfigurasi preprocessing dan augmentasi
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import yaml
import sys
from pathlib import Path

# Import modul preprocessing
from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import (
    ensure_ui_persistence as ensure_preprocessing_persistence,
    get_preprocessing_config,
    sync_config_with_drive as sync_preprocessing_config,
    reset_config_to_default as reset_preprocessing_config
)

# Import modul augmentasi
from smartcash.ui.dataset.augmentation.handlers.config_persistence import (
    ensure_ui_persistence as ensure_augmentation_persistence,
    get_augmentation_config,
    sync_config_with_drive as sync_augmentation_config,
    reset_config_to_default as reset_augmentation_config
)

# Import fungsi utilitas
from smartcash.ui.dataset.preprocessing.tests.test_utils import (
    setup_test_environment,
    restore_environment,
    close_all_loggers
)

class TestConfigIntegration(unittest.TestCase):
    """Kelas pengujian untuk integrasi persistensi konfigurasi"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Siapkan lingkungan pengujian
        setup_test_environment()
        
        # Mock UI components untuk preprocessing
        self.mock_preprocessing_components = {
            'status': MagicMock(),
            'logger': MagicMock(),
            'preprocess_options': MagicMock(),
            'validation_options': MagicMock(),
            'split_selector': MagicMock(),
            'config': {
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
        }
        
        # Mock UI components untuk augmentasi
        self.mock_augmentation_components = {
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
                    'dataset_path': 'data/preprocessed'
                }
            }
        }
        
        # Mock ConfigManager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.side_effect = lambda module, default=None: (
            self.mock_preprocessing_components['config'] if module == 'preprocessing' 
            else self.mock_augmentation_components['config']
        )
        self.mock_config_manager.save_module_config.return_value = True
        self.mock_config_manager.register_ui_components = MagicMock()
        self.mock_config_manager.get_ui_components.side_effect = lambda module: (
            self.mock_preprocessing_components if module == 'preprocessing' 
            else self.mock_augmentation_components
        )
        
        # Patch untuk get_config_manager
        self.patcher_config_manager = patch(
            'smartcash.common.config.get_config_manager',
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
        self.patcher_yaml_load = patch('yaml.safe_load', side_effect=lambda f: (
            self.mock_preprocessing_components['config'] if 'preprocessing' in str(f) 
            else self.mock_augmentation_components['config']
        ))
        self.mock_yaml_load = self.patcher_yaml_load.start()
        
        # Patch untuk os.path.exists
        self.patcher_exists = patch('os.path.exists', return_value=True)
        self.mock_exists = self.patcher_exists.start()
        
        # Patch untuk environment manager
        self.mock_env_manager = MagicMock()
        self.mock_env_manager.is_drive_mounted = True
        self.mock_env_manager.drive_path = Path('/mock/drive/path')
        
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
        self.patcher_env_manager.stop()
        
        # Tutup semua logger untuk menghindari ResourceWarning
        close_all_loggers()
        
        # Kembalikan lingkungan pengujian ke keadaan semula
        restore_environment()

    def test_preprocessing_to_augmentation_path_integration(self):
        """Pengujian integrasi jalur data antara preprocessing dan augmentasi"""
        # Verifikasi bahwa jalur output preprocessing adalah jalur input augmentasi
        self.assertEqual(
            self.mock_preprocessing_components['config']['preprocessing']['output_dir'],
            self.mock_augmentation_components['config']['data']['dataset_path']
        )
    
    def test_reset_both_configs(self):
        """Pengujian reset konfigurasi preprocessing dan augmentasi"""
        # Reset konfigurasi preprocessing
        preprocessing_result = reset_preprocessing_config(self.mock_preprocessing_components)
        
        # Reset konfigurasi augmentasi
        augmentation_result = reset_augmentation_config(self.mock_augmentation_components)
        
        # Verifikasi hasil
        self.assertTrue(preprocessing_result)
        self.assertTrue(augmentation_result)
        
        # Verifikasi bahwa konfigurasi diatur ke nilai default
        self.assertIsNotNone(self.mock_preprocessing_components.get('config'))
        self.assertIsNotNone(self.mock_augmentation_components.get('config'))
    
    def test_sync_both_configs(self):
        """Pengujian sinkronisasi konfigurasi preprocessing dan augmentasi"""
        # Sinkronisasi konfigurasi preprocessing
        preprocessing_result = sync_preprocessing_config(self.mock_preprocessing_components)
        
        # Sinkronisasi konfigurasi augmentasi
        augmentation_result = sync_augmentation_config(self.mock_augmentation_components)
        
        # Verifikasi hasil
        self.assertTrue(preprocessing_result)
        self.assertTrue(augmentation_result)
        
        # Verifikasi bahwa konfigurasi tersedia
        self.assertIsNotNone(self.mock_preprocessing_components.get('config'))
        self.assertIsNotNone(self.mock_augmentation_components.get('config'))
    
    def test_ensure_persistence_for_both(self):
        """Pengujian ensure_ui_persistence untuk preprocessing dan augmentasi"""
        # Panggil fungsi ensure_ui_persistence untuk preprocessing
        preprocessing_result = ensure_preprocessing_persistence(self.mock_preprocessing_components)
        
        # Panggil fungsi ensure_ui_persistence untuk augmentasi
        augmentation_result = ensure_augmentation_persistence(self.mock_augmentation_components)
        
        # Verifikasi hasil
        self.assertEqual(preprocessing_result, self.mock_preprocessing_components)
        self.assertEqual(augmentation_result, self.mock_augmentation_components)
        
        # Verifikasi bahwa UI components memiliki konfigurasi
        self.assertIsNotNone(preprocessing_result.get('config'))
        self.assertIsNotNone(augmentation_result.get('config'))

if __name__ == '__main__':
    unittest.main()

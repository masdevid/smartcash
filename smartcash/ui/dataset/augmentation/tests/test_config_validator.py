"""
File: smartcash/ui/dataset/augmentation/tests/test_config_validator.py
Deskripsi: Pengujian untuk validator konfigurasi augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import os

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.config_validator import (
    validate_augmentation_config,
    _validate_param,
    _validate_range_param,
    _validate_aug_types
)

class TestConfigValidator(unittest.TestCase):
    """Kelas pengujian untuk config_validator.py"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Mock konfigurasi valid
        self.valid_config = {
            'augmentation': {
                'types': ['Combined (Recommended)'],
                'prefix': 'aug_',
                'num_variations': 2,
                'balance_classes': False,
                'num_workers': 4,
                'position': {
                    'fliplr': 0.5,
                    'degrees': 15
                },
                'lighting': {
                    'hsv_h': 0.025,
                    'contrast': [0.7, 1.3]
                }
            }
        }
        
        # Mock konfigurasi tidak valid
        self.invalid_config = {
            'augmentation': {
                'types': None,
                'prefix': '',
                'num_variations': -1,
                'balance_classes': False,
                'num_workers': 0
            }
        }

    def test_validate_param_valid(self):
        """Pengujian _validate_param dengan nilai valid"""
        # Panggil fungsi yang diuji
        result = _validate_param(5, 2, int, (1, 10))
        
        # Verifikasi hasil
        self.assertEqual(result, 5)

    def test_validate_param_invalid_type(self):
        """Pengujian _validate_param dengan tipe tidak valid"""
        # Panggil fungsi yang diuji
        result = _validate_param("abc", 2, int)
        
        # Verifikasi hasil
        self.assertEqual(result, 2)

    def test_validate_param_out_of_range(self):
        """Pengujian _validate_param dengan nilai di luar rentang"""
        # Panggil fungsi yang diuji
        result = _validate_param(15, 5, int, (1, 10))
        
        # Verifikasi hasil
        self.assertEqual(result, 5)

    def test_validate_range_param_valid(self):
        """Pengujian _validate_range_param dengan nilai valid"""
        # Panggil fungsi yang diuji
        result = _validate_range_param([0.5, 1.5], [0.7, 1.3])
        
        # Verifikasi hasil
        self.assertEqual(result, [0.5, 1.5])

    def test_validate_range_param_invalid(self):
        """Pengujian _validate_range_param dengan nilai tidak valid"""
        # Panggil fungsi yang diuji
        result = _validate_range_param("not_a_list", [0.7, 1.3])
        
        # Verifikasi hasil
        self.assertEqual(result, [0.7, 1.3])

    def test_validate_aug_types_valid(self):
        """Pengujian _validate_aug_types dengan nilai valid"""
        # Panggil fungsi yang diuji
        result = _validate_aug_types(['Combined (Recommended)'])
        
        # Verifikasi hasil
        self.assertEqual(result, ['combined'])

    def test_validate_aug_types_invalid(self):
        """Pengujian _validate_aug_types dengan nilai tidak valid"""
        # Panggil fungsi yang diuji
        result = _validate_aug_types(None)
        
        # Verifikasi hasil
        self.assertEqual(result, ['combined'])

    def test_validate_augmentation_config(self):
        """Pengujian validate_augmentation_config"""
        # Panggil fungsi yang diuji
        result = validate_augmentation_config(self.valid_config)
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('augmentation', result)
        self.assertIn('types', result['augmentation'])
        self.assertIn('position', result['augmentation'])
        self.assertIn('lighting', result['augmentation'])

    def test_validate_augmentation_config_empty(self):
        """Pengujian validate_augmentation_config dengan config kosong"""
        # Panggil fungsi yang diuji
        result = validate_augmentation_config({})
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('augmentation', result)
        self.assertIn('types', result['augmentation'])
        self.assertEqual(result['augmentation']['types'], ['combined'])

    def test_validate_augmentation_config_invalid(self):
        """Pengujian validate_augmentation_config dengan nilai tidak valid"""
        # Panggil fungsi yang diuji
        result = validate_augmentation_config(self.invalid_config)
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('augmentation', result)
        self.assertEqual(result['augmentation']['num_variations'], 2)  # Default karena -1 tidak valid

if __name__ == '__main__':
    unittest.main()

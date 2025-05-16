"""
File: smartcash/ui/dataset/augmentation/tests/test_parameter_handler.py
Deskripsi: Test untuk parameter handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.dataset.augmentation.handlers.parameter_handler import (
    validate_augmentation_params,
    map_ui_to_config,
    map_config_to_ui
)

class TestParameterHandler(unittest.TestCase):
    """Test untuk parameter handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'data_dir': 'data',
            'split_selector': MagicMock()
        }
        
        # Mock untuk config handler
        self.patcher1 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.get_config_from_ui')
        self.mock_get_config = self.patcher1.start()
        
        self.patcher2 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.update_config_from_ui')
        self.mock_update_config = self.patcher2.start()
        
        self.patcher3 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.update_ui_from_config')
        self.mock_update_ui = self.patcher3.start()
        
        # Mock untuk os.path
        self.patcher4 = patch('os.path.exists')
        self.mock_exists = self.patcher4.start()
        self.mock_exists.return_value = True
        
        self.patcher5 = patch('os.path.join')
        self.mock_join = self.patcher5.start()
        self.mock_join.side_effect = lambda *args: '/'.join(args)
        
        self.patcher6 = patch('os.listdir')
        self.mock_listdir = self.patcher6.start()
        self.mock_listdir.return_value = ['file1.jpg', 'file2.jpg']
        
        # Setup mock split selector
        mock_dropdown = MagicMock()
        mock_dropdown.description = 'Split:'
        mock_dropdown.value = 'train'
        
        mock_child = MagicMock()
        mock_child.children = [mock_dropdown]
        
        self.ui_components['split_selector'].children = [mock_child]
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        self.patcher6.stop()
    
    def test_validate_augmentation_params_valid(self):
        """Test validasi parameter augmentasi yang valid"""
        # Setup valid config
        valid_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'output_prefix': 'aug',
                'position': {
                    'fliplr': 0.5,
                    'degrees': 15,
                    'translate': 0.1,
                    'scale': 0.1,
                    'shear_max': 10
                },
                'lighting': {
                    'hsv_h': 0.1,
                    'hsv_s': 0.5,
                    'hsv_v': 0.5,
                    'contrast': [0.8, 1.2],
                    'brightness': [0.8, 1.2],
                    'blur': 0.1,
                    'noise': 0.1
                }
            }
        }
        self.mock_get_config.return_value = valid_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['message'], 'Parameter augmentasi valid')
    
    def test_validate_augmentation_params_disabled(self):
        """Test validasi parameter augmentasi yang tidak diaktifkan"""
        # Setup disabled config
        disabled_config = {
            'augmentation': {
                'enabled': False
            }
        }
        self.mock_get_config.return_value = disabled_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Augmentasi tidak diaktifkan')
    
    def test_validate_augmentation_params_invalid_types(self):
        """Test validasi parameter augmentasi dengan jenis yang tidak valid"""
        # Setup config with invalid types
        invalid_types_config = {
            'augmentation': {
                'enabled': True,
                'types': []
            }
        }
        self.mock_get_config.return_value = invalid_types_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Jenis augmentasi tidak valid atau tidak dipilih')
    
    def test_validate_augmentation_params_invalid_num_variations(self):
        """Test validasi parameter augmentasi dengan jumlah variasi tidak valid"""
        # Setup config with invalid num_variations
        invalid_num_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 0
            }
        }
        self.mock_get_config.return_value = invalid_num_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Jumlah variasi harus lebih dari 0')
    
    def test_validate_augmentation_params_invalid_target_count(self):
        """Test validasi parameter augmentasi dengan target count tidak valid"""
        # Setup config with invalid target_count
        invalid_target_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 0
            }
        }
        self.mock_get_config.return_value = invalid_target_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Target jumlah per kelas harus lebih dari 0')
    
    def test_validate_augmentation_params_invalid_output_prefix(self):
        """Test validasi parameter augmentasi dengan output prefix tidak valid"""
        # Setup config with invalid output_prefix
        invalid_prefix_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'output_prefix': ''
            }
        }
        self.mock_get_config.return_value = invalid_prefix_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Output prefix tidak boleh kosong')
    
    def test_validate_augmentation_params_invalid_position_params(self):
        """Test validasi parameter augmentasi dengan parameter posisi tidak valid"""
        # Setup config with invalid position params
        invalid_position_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'output_prefix': 'aug',
                'position': {
                    'fliplr': -0.5,  # Invalid value
                    'degrees': 15,
                    'translate': 0.1,
                    'scale': 0.1,
                    'shear_max': 10
                }
            }
        }
        self.mock_get_config.return_value = invalid_position_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Probabilitas flip horizontal harus antara 0 dan 1')
    
    def test_validate_augmentation_params_invalid_lighting_params(self):
        """Test validasi parameter augmentasi dengan parameter pencahayaan tidak valid"""
        # Setup config with invalid lighting params
        invalid_lighting_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'output_prefix': 'aug',
                'position': {
                    'fliplr': 0.5,
                    'degrees': 15,
                    'translate': 0.1,
                    'scale': 0.1,
                    'shear_max': 10
                },
                'lighting': {
                    'hsv_h': 1.5,  # Invalid value
                    'hsv_s': 0.5,
                    'hsv_v': 0.5,
                    'contrast': [0.8, 1.2],
                    'brightness': [0.8, 1.2],
                    'blur': 0.1,
                    'noise': 0.1
                }
            }
        }
        self.mock_get_config.return_value = invalid_lighting_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'HSV Hue harus antara 0 dan 1')
    
    def test_validate_augmentation_params_invalid_contrast(self):
        """Test validasi parameter augmentasi dengan contrast tidak valid"""
        # Setup config with invalid contrast
        invalid_contrast_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'output_prefix': 'aug',
                'position': {
                    'fliplr': 0.5,
                    'degrees': 15,
                    'translate': 0.1,
                    'scale': 0.1,
                    'shear_max': 10
                },
                'lighting': {
                    'hsv_h': 0.1,
                    'hsv_s': 0.5,
                    'hsv_v': 0.5,
                    'contrast': [1.2, 0.8],  # Invalid order
                    'brightness': [0.8, 1.2],
                    'blur': 0.1,
                    'noise': 0.1
                }
            }
        }
        self.mock_get_config.return_value = invalid_contrast_config
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'Range contrast tidak valid')
    
    def test_validate_augmentation_params_dataset_not_found(self):
        """Test validasi parameter augmentasi dengan dataset tidak ditemukan"""
        # Setup valid config
        valid_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'output_prefix': 'aug',
                'position': {
                    'fliplr': 0.5,
                    'degrees': 15,
                    'translate': 0.1,
                    'scale': 0.1,
                    'shear_max': 10
                },
                'lighting': {
                    'hsv_h': 0.1,
                    'hsv_s': 0.5,
                    'hsv_v': 0.5,
                    'contrast': [0.8, 1.2],
                    'brightness': [0.8, 1.2],
                    'blur': 0.1,
                    'noise': 0.1
                }
            }
        }
        self.mock_get_config.return_value = valid_config
        
        # Mock path.exists untuk mengembalikan False untuk dataset
        self.mock_exists.side_effect = lambda path: False if 'preprocessed' in path else True
        
        # Panggil fungsi
        result = validate_augmentation_params(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result['status'], 'error')
        self.assertTrue('Dataset' in result['message'] and 'tidak ditemukan' in result['message'])
    
    def test_map_ui_to_config(self):
        """Test pemetaan UI ke konfigurasi"""
        # Setup
        expected_config = {'augmentation': {'enabled': True}}
        self.mock_update_config.return_value = expected_config
        
        # Panggil fungsi
        result = map_ui_to_config(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, expected_config)
        self.mock_update_config.assert_called_once_with(self.ui_components, None)
    
    def test_map_config_to_ui(self):
        """Test pemetaan konfigurasi ke UI"""
        # Setup
        config = {'augmentation': {'enabled': True}}
        
        # Panggil fungsi
        map_config_to_ui(self.ui_components, config)
        
        # Verifikasi hasil
        self.mock_update_ui.assert_called_once_with(self.ui_components, config)

if __name__ == '__main__':
    unittest.main()

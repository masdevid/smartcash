"""
File: smartcash/tests/test_training_strategy_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config training strategy dengan training_config.yaml
"""

import unittest
import yaml
import os
from pathlib import Path
from typing import Dict, Any

from smartcash.ui.training_config.training_strategy.handlers.defaults import get_default_training_strategy_config
from smartcash.ui.training_config.training_strategy.handlers.ui_extractor import extract_training_strategy_config
from smartcash.ui.training_config.training_strategy.handlers.ui_updater import update_training_strategy_ui


class TestTrainingStrategyConfig(unittest.TestCase):
    """Test kesesuaian struktur config training strategy dengan training_config.yaml"""

    def setUp(self):
        """Setup untuk test dengan membaca file training_config.yaml"""
        # Path ke file training_config.yaml
        config_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "configs" / "training_config.yaml"
        
        # Baca file training_config.yaml
        with open(config_path, 'r') as file:
            self.yaml_config = yaml.safe_load(file)
        
        # Dapatkan default config dari handler
        self.default_config = get_default_training_strategy_config()
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            # Parameter validasi
            'validation_frequency_slider': type('obj', (object,), {'value': 1}),
            'iou_threshold_slider': type('obj', (object,), {'value': 0.6}),
            'conf_threshold_slider': type('obj', (object,), {'value': 0.001}),
            
            # Parameter multi-scale
            'multi_scale_checkbox': type('obj', (object,), {'value': True}),
            
            # Parameter training utils
            'experiment_name_text': type('obj', (object,), {'value': 'efficientnet_b4_training'}),
            'checkpoint_dir_text': type('obj', (object,), {'value': '/content/runs/train/checkpoints'}),
            'tensorboard_checkbox': type('obj', (object,), {'value': True}),
            'log_metrics_every_slider': type('obj', (object,), {'value': 10}),
            'visualize_batch_every_slider': type('obj', (object,), {'value': 100}),
            'gradient_clipping_slider': type('obj', (object,), {'value': 1.0}),
            'mixed_precision_checkbox': type('obj', (object,), {'value': True}),
            'layer_mode_dropdown': type('obj', (object,), {'value': 'single'}),
            
            # Status panel untuk logging
            'status_panel': type('obj', (object,), {'clear_output': lambda: None})
        }
        
        # Extract config dari mock UI components
        self.extracted_config = extract_training_strategy_config(self.mock_ui_components)

    def test_default_config_structure(self):
        """Test struktur default config sesuai dengan training_config.yaml"""
        # Verifikasi parameter validasi
        self.assertIn('validation', self.default_config)
        self.assertIsInstance(self.default_config['validation'], dict)
        self.assertIn('frequency', self.default_config['validation'])
        self.assertIn('iou_thres', self.default_config['validation'])
        self.assertIn('conf_thres', self.default_config['validation'])
        
        # Verifikasi parameter multi-scale
        self.assertIn('multi_scale', self.default_config)
        self.assertIsInstance(self.default_config['multi_scale'], bool)
        
        # Verifikasi parameter training utils
        self.assertIn('training_utils', self.default_config)
        self.assertIsInstance(self.default_config['training_utils'], dict)
        self.assertIn('experiment_name', self.default_config['training_utils'])
        self.assertIn('checkpoint_dir', self.default_config['training_utils'])
        self.assertIn('tensorboard', self.default_config['training_utils'])
        self.assertIn('log_metrics_every', self.default_config['training_utils'])
        self.assertIn('visualize_batch_every', self.default_config['training_utils'])
        self.assertIn('gradient_clipping', self.default_config['training_utils'])
        self.assertIn('mixed_precision', self.default_config['training_utils'])
        self.assertIn('layer_mode', self.default_config['training_utils'])
        
        # Verifikasi tipe data untuk beberapa field penting
        self.assertIsInstance(self.default_config['validation']['frequency'], int)
        self.assertIsInstance(self.default_config['validation']['iou_thres'], float)
        self.assertIsInstance(self.default_config['validation']['conf_thres'], float)
        self.assertIsInstance(self.default_config['multi_scale'], bool)
        self.assertIsInstance(self.default_config['training_utils']['experiment_name'], str)
        self.assertIsInstance(self.default_config['training_utils']['tensorboard'], bool)
        self.assertIsInstance(self.default_config['training_utils']['log_metrics_every'], int)

    def test_yaml_config_structure(self):
        """Test struktur YAML config sesuai dengan yang diharapkan"""
        # Verifikasi parameter validasi
        self.assertIn('validation', self.yaml_config)
        self.assertIsInstance(self.yaml_config['validation'], dict)
        self.assertIn('frequency', self.yaml_config['validation'])
        self.assertIn('iou_thres', self.yaml_config['validation'])
        self.assertIn('conf_thres', self.yaml_config['validation'])
        
        # Verifikasi parameter multi-scale
        self.assertIn('multi_scale', self.yaml_config)
        
        # Verifikasi parameter training utils
        self.assertIn('training_utils', self.yaml_config)
        self.assertIsInstance(self.yaml_config['training_utils'], dict)
        self.assertIn('experiment_name', self.yaml_config['training_utils'])
        self.assertIn('checkpoint_dir', self.yaml_config['training_utils'])
        self.assertIn('tensorboard', self.yaml_config['training_utils'])
        self.assertIn('log_metrics_every', self.yaml_config['training_utils'])
        self.assertIn('visualize_batch_every', self.yaml_config['training_utils'])
        self.assertIn('gradient_clipping', self.yaml_config['training_utils'])
        self.assertIn('mixed_precision', self.yaml_config['training_utils'])
        self.assertIn('layer_mode', self.yaml_config['training_utils'])

    def test_extracted_config_structure(self):
        """Test struktur extracted config sesuai dengan training_config.yaml"""
        # Verifikasi parameter validasi
        self.assertIn('validation', self.extracted_config)
        self.assertIsInstance(self.extracted_config['validation'], dict)
        self.assertIn('frequency', self.extracted_config['validation'])
        self.assertIn('iou_thres', self.extracted_config['validation'])
        self.assertIn('conf_thres', self.extracted_config['validation'])
        
        # Verifikasi parameter multi-scale
        self.assertIn('multi_scale', self.extracted_config)
        self.assertIsInstance(self.extracted_config['multi_scale'], bool)
        
        # Verifikasi parameter training utils
        self.assertIn('training_utils', self.extracted_config)
        self.assertIsInstance(self.extracted_config['training_utils'], dict)
        self.assertIn('experiment_name', self.extracted_config['training_utils'])
        self.assertIn('checkpoint_dir', self.extracted_config['training_utils'])
        self.assertIn('tensorboard', self.extracted_config['training_utils'])
        self.assertIn('log_metrics_every', self.extracted_config['training_utils'])
        self.assertIn('visualize_batch_every', self.extracted_config['training_utils'])
        self.assertIn('gradient_clipping', self.extracted_config['training_utils'])
        self.assertIn('mixed_precision', self.extracted_config['training_utils'])
        self.assertIn('layer_mode', self.extracted_config['training_utils'])
        
        # Verifikasi nilai yang diambil dari UI components
        self.assertEqual(self.extracted_config['validation']['frequency'], 1)
        self.assertEqual(self.extracted_config['validation']['iou_thres'], 0.6)
        self.assertEqual(self.extracted_config['validation']['conf_thres'], 0.001)
        self.assertEqual(self.extracted_config['multi_scale'], True)
        self.assertEqual(self.extracted_config['training_utils']['experiment_name'], 'efficientnet_b4_training')
        self.assertEqual(self.extracted_config['training_utils']['tensorboard'], True)
        self.assertEqual(self.extracted_config['training_utils']['log_metrics_every'], 10)
        self.assertEqual(self.extracted_config['training_utils']['layer_mode'], 'single')

    def test_ui_updater(self):
        """Test UI updater dapat mengupdate UI components dengan benar"""
        # Buat mock UI components baru untuk testing
        mock_ui_for_update = {
            # Parameter validasi
            'validation_frequency_slider': type('obj', (object,), {'value': 2}),
            'iou_threshold_slider': type('obj', (object,), {'value': 0.5}),
            'conf_threshold_slider': type('obj', (object,), {'value': 0.01}),
            
            # Parameter multi-scale
            'multi_scale_checkbox': type('obj', (object,), {'value': False}),
            
            # Parameter training utils
            'experiment_name_text': type('obj', (object,), {'value': 'test_training'}),
            'checkpoint_dir_text': type('obj', (object,), {'value': '/content/test/checkpoints'}),
            'tensorboard_checkbox': type('obj', (object,), {'value': False}),
            'log_metrics_every_slider': type('obj', (object,), {'value': 20}),
            'visualize_batch_every_slider': type('obj', (object,), {'value': 200}),
            'gradient_clipping_slider': type('obj', (object,), {'value': 0.5}),
            'mixed_precision_checkbox': type('obj', (object,), {'value': False}),
            'layer_mode_dropdown': type('obj', (object,), {'value': 'multilayer'}),
            
            # Status panel untuk logging
            'status_panel': type('obj', (object,), {'clear_output': lambda: None})
        }
        
        # Buat config untuk update
        update_config = {
            'validation': {
                'frequency': 1,
                'iou_thres': 0.6,
                'conf_thres': 0.001
            },
            'multi_scale': True,
            'training_utils': {
                'experiment_name': 'efficientnet_b4_training',
                'checkpoint_dir': '/content/runs/train/checkpoints',
                'tensorboard': True,
                'log_metrics_every': 10,
                'visualize_batch_every': 100,
                'gradient_clipping': 1.0,
                'mixed_precision': True,
                'layer_mode': 'single'
            }
        }
        
        # Update UI components
        update_training_strategy_ui(mock_ui_for_update, update_config)
        
        # Verifikasi bahwa UI components telah diupdate dengan benar
        self.assertEqual(mock_ui_for_update['validation_frequency_slider'].value, 1)
        self.assertEqual(mock_ui_for_update['iou_threshold_slider'].value, 0.6)
        self.assertEqual(mock_ui_for_update['conf_threshold_slider'].value, 0.001)
        self.assertEqual(mock_ui_for_update['multi_scale_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['experiment_name_text'].value, 'efficientnet_b4_training')
        self.assertEqual(mock_ui_for_update['checkpoint_dir_text'].value, '/content/runs/train/checkpoints')
        self.assertEqual(mock_ui_for_update['tensorboard_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['log_metrics_every_slider'].value, 10)
        self.assertEqual(mock_ui_for_update['visualize_batch_every_slider'].value, 100)
        self.assertEqual(mock_ui_for_update['gradient_clipping_slider'].value, 1.0)
        self.assertEqual(mock_ui_for_update['mixed_precision_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['layer_mode_dropdown'].value, 'single')

    def test_default_vs_yaml_config(self):
        """Test kesesuaian default config dengan YAML config"""
        # Dapatkan key dari default config dan YAML config
        default_keys = set(self.default_config.keys())
        yaml_keys = set(self.yaml_config.keys())
        
        # Hapus key metadata yang mungkin berbeda
        default_keys.discard('config_version')
        default_keys.discard('description')
        yaml_keys.discard('_base_')
        yaml_keys.discard('updated_at')
        
        # Verifikasi bahwa semua key di default config ada di YAML config
        missing_in_yaml = default_keys - yaml_keys
        self.assertEqual(len(missing_in_yaml), 0, f"Key berikut tidak ada di YAML config: {missing_in_yaml}")
        
        # Verifikasi bahwa semua key di YAML config ada di default config
        missing_in_default = yaml_keys - default_keys
        self.assertEqual(len(missing_in_default), 0, f"Key berikut tidak ada di default config: {missing_in_default}")
        
        # Verifikasi struktur nested untuk validation
        self.assertIn('validation', self.default_config)
        self.assertIn('validation', self.yaml_config)
        self.assertIsInstance(self.default_config['validation'], dict)
        self.assertIsInstance(self.yaml_config['validation'], dict)
        
        default_validation_keys = set(self.default_config['validation'].keys())
        yaml_validation_keys = set(self.yaml_config['validation'].keys())
        
        self.assertEqual(default_validation_keys, yaml_validation_keys)
        
        # Verifikasi struktur nested untuk training_utils
        self.assertIn('training_utils', self.default_config)
        self.assertIn('training_utils', self.yaml_config)
        self.assertIsInstance(self.default_config['training_utils'], dict)
        self.assertIsInstance(self.yaml_config['training_utils'], dict)
        
        default_training_utils_keys = set(self.default_config['training_utils'].keys())
        yaml_training_utils_keys = set(self.yaml_config['training_utils'].keys())
        
        self.assertEqual(default_training_utils_keys, yaml_training_utils_keys)


if __name__ == '__main__':
    unittest.main()

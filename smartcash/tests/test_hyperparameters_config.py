"""
File: smartcash/tests/test_hyperparameters_config.py
Deskripsi: Test untuk memverifikasi kesesuaian struktur config hyperparameters dengan hyperparameters_config.yaml
"""

import unittest
import yaml
import os
from pathlib import Path
from typing import Dict, Any

from smartcash.ui.training_config.hyperparameters.handlers.defaults import get_default_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_extractor import extract_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_updater import update_hyperparameters_ui


class TestHyperparametersConfig(unittest.TestCase):
    """Test kesesuaian struktur config hyperparameters dengan hyperparameters_config.yaml"""

    def setUp(self):
        """Setup untuk test dengan membaca file hyperparameters_config.yaml"""
        # Path ke file hyperparameters_config.yaml
        config_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "configs" / "hyperparameters_config.yaml"
        
        # Baca file hyperparameters_config.yaml
        with open(config_path, 'r') as file:
            self.yaml_config = yaml.safe_load(file)
        
        # Dapatkan default config dari handler
        self.default_config = get_default_hyperparameters_config()
        
        # Mock UI components untuk testing
        self.mock_ui_components = {
            # Parameter dasar
            'batch_size_slider': type('obj', (object,), {'value': 16}),
            'image_size_slider': type('obj', (object,), {'value': 640}),
            'epochs_slider': type('obj', (object,), {'value': 100}),
            'dropout_slider': type('obj', (object,), {'value': 0.0}),
            
            # Parameter optimasi
            'optimizer_dropdown': type('obj', (object,), {'value': 'Adam'}),
            'learning_rate_slider': type('obj', (object,), {'value': 0.01}),
            'weight_decay_slider': type('obj', (object,), {'value': 0.0005}),
            'momentum_slider': type('obj', (object,), {'value': 0.937}),
            
            # Parameter penjadwalan
            'scheduler_dropdown': type('obj', (object,), {'value': 'cosine'}),
            'warmup_epochs_slider': type('obj', (object,), {'value': 3}),
            
            # Parameter regularisasi
            'augment_checkbox': type('obj', (object,), {'value': True}),
            
            # Parameter loss
            'box_loss_gain_slider': type('obj', (object,), {'value': 0.05}),
            'cls_loss_gain_slider': type('obj', (object,), {'value': 0.5}),
            'obj_loss_gain_slider': type('obj', (object,), {'value': 1.0}),
            
            # Parameter early stopping
            'early_stopping_checkbox': type('obj', (object,), {'value': True}),
            'patience_slider': type('obj', (object,), {'value': 15}),
            'min_delta_slider': type('obj', (object,), {'value': 0.001}),
            
            # Parameter save best
            'save_best_checkbox': type('obj', (object,), {'value': True}),
            'checkpoint_metric_dropdown': type('obj', (object,), {'value': 'mAP_0.5'}),
            
            # Status panel untuk logging
            'status_panel': type('obj', (object,), {'clear_output': lambda: None})
        }
        
        # Extract config dari mock UI components
        self.extracted_config = extract_hyperparameters_config(self.mock_ui_components)

    def test_default_config_structure(self):
        """Test struktur default config sesuai dengan hyperparameters_config.yaml"""
        # Verifikasi parameter dasar
        self.assertIn('batch_size', self.default_config)
        self.assertIn('image_size', self.default_config)
        self.assertIn('epochs', self.default_config)
        
        # Verifikasi parameter optimasi
        self.assertIn('optimizer', self.default_config)
        self.assertIn('learning_rate', self.default_config)
        self.assertIn('weight_decay', self.default_config)
        self.assertIn('momentum', self.default_config)
        
        # Verifikasi parameter penjadwalan
        self.assertIn('scheduler', self.default_config)
        self.assertIn('warmup_epochs', self.default_config)
        self.assertIn('warmup_momentum', self.default_config)
        self.assertIn('warmup_bias_lr', self.default_config)
        
        # Verifikasi parameter regularisasi
        self.assertIn('augment', self.default_config)
        self.assertIn('dropout', self.default_config)
        
        # Verifikasi parameter loss
        self.assertIn('box_loss_gain', self.default_config)
        self.assertIn('cls_loss_gain', self.default_config)
        self.assertIn('obj_loss_gain', self.default_config)
        
        # Verifikasi parameter anchor
        self.assertIn('anchor_t', self.default_config)
        self.assertIn('fl_gamma', self.default_config)
        
        # Verifikasi parameter early stopping
        self.assertIn('early_stopping', self.default_config)
        self.assertIsInstance(self.default_config['early_stopping'], dict)
        self.assertIn('enabled', self.default_config['early_stopping'])
        self.assertIn('patience', self.default_config['early_stopping'])
        self.assertIn('min_delta', self.default_config['early_stopping'])
        
        # Verifikasi parameter save best
        self.assertIn('save_best', self.default_config)
        self.assertIsInstance(self.default_config['save_best'], dict)
        self.assertIn('enabled', self.default_config['save_best'])
        self.assertIn('metric', self.default_config['save_best'])
        
        # Verifikasi tipe data untuk beberapa field penting
        self.assertIsInstance(self.default_config['batch_size'], int)
        self.assertIsInstance(self.default_config['image_size'], int)
        self.assertIsInstance(self.default_config['epochs'], int)
        self.assertIsInstance(self.default_config['optimizer'], str)
        self.assertIsInstance(self.default_config['learning_rate'], float)
        self.assertIsInstance(self.default_config['augment'], bool)

    def test_yaml_config_structure(self):
        """Test struktur YAML config sesuai dengan yang diharapkan"""
        # Verifikasi parameter dasar
        self.assertIn('batch_size', self.yaml_config)
        self.assertIn('image_size', self.yaml_config)
        self.assertIn('epochs', self.yaml_config)
        
        # Verifikasi parameter optimasi
        self.assertIn('optimizer', self.yaml_config)
        self.assertIn('learning_rate', self.yaml_config)
        self.assertIn('weight_decay', self.yaml_config)
        self.assertIn('momentum', self.yaml_config)
        
        # Verifikasi parameter penjadwalan
        self.assertIn('scheduler', self.yaml_config)
        self.assertIn('warmup_epochs', self.yaml_config)
        self.assertIn('warmup_momentum', self.yaml_config)
        self.assertIn('warmup_bias_lr', self.yaml_config)
        
        # Verifikasi parameter regularisasi
        self.assertIn('augment', self.yaml_config)
        self.assertIn('dropout', self.yaml_config)
        
        # Verifikasi parameter loss
        self.assertIn('box_loss_gain', self.yaml_config)
        self.assertIn('cls_loss_gain', self.yaml_config)
        self.assertIn('obj_loss_gain', self.yaml_config)
        
        # Verifikasi parameter anchor
        self.assertIn('anchor_t', self.yaml_config)
        self.assertIn('fl_gamma', self.yaml_config)
        
        # Verifikasi parameter early stopping
        self.assertIn('early_stopping', self.yaml_config)
        self.assertIsInstance(self.yaml_config['early_stopping'], dict)
        self.assertIn('enabled', self.yaml_config['early_stopping'])
        self.assertIn('patience', self.yaml_config['early_stopping'])
        self.assertIn('min_delta', self.yaml_config['early_stopping'])
        
        # Verifikasi parameter save best
        self.assertIn('save_best', self.yaml_config)
        self.assertIsInstance(self.yaml_config['save_best'], dict)
        self.assertIn('enabled', self.yaml_config['save_best'])
        self.assertIn('metric', self.yaml_config['save_best'])

    def test_extracted_config_structure(self):
        """Test struktur extracted config sesuai dengan hyperparameters_config.yaml"""
        # Verifikasi parameter dasar
        self.assertIn('batch_size', self.extracted_config)
        self.assertIn('image_size', self.extracted_config)
        self.assertIn('epochs', self.extracted_config)
        
        # Verifikasi parameter optimasi
        self.assertIn('optimizer', self.extracted_config)
        self.assertIn('learning_rate', self.extracted_config)
        self.assertIn('weight_decay', self.extracted_config)
        self.assertIn('momentum', self.extracted_config)
        
        # Verifikasi parameter penjadwalan
        self.assertIn('scheduler', self.extracted_config)
        self.assertIn('warmup_epochs', self.extracted_config)
        self.assertIn('warmup_momentum', self.extracted_config)
        self.assertIn('warmup_bias_lr', self.extracted_config)
        
        # Verifikasi parameter regularisasi
        self.assertIn('augment', self.extracted_config)
        self.assertIn('dropout', self.extracted_config)
        
        # Verifikasi parameter loss
        self.assertIn('box_loss_gain', self.extracted_config)
        self.assertIn('cls_loss_gain', self.extracted_config)
        self.assertIn('obj_loss_gain', self.extracted_config)
        
        # Verifikasi parameter anchor
        self.assertIn('anchor_t', self.extracted_config)
        self.assertIn('fl_gamma', self.extracted_config)
        
        # Verifikasi parameter early stopping
        self.assertIn('early_stopping', self.extracted_config)
        self.assertIsInstance(self.extracted_config['early_stopping'], dict)
        self.assertIn('enabled', self.extracted_config['early_stopping'])
        self.assertIn('patience', self.extracted_config['early_stopping'])
        self.assertIn('min_delta', self.extracted_config['early_stopping'])
        
        # Verifikasi parameter save best
        self.assertIn('save_best', self.extracted_config)
        self.assertIsInstance(self.extracted_config['save_best'], dict)
        self.assertIn('enabled', self.extracted_config['save_best'])
        self.assertIn('metric', self.extracted_config['save_best'])
        
        # Verifikasi nilai yang diambil dari UI components
        self.assertEqual(self.extracted_config['batch_size'], 16)
        self.assertEqual(self.extracted_config['image_size'], 640)
        self.assertEqual(self.extracted_config['epochs'], 100)
        self.assertEqual(self.extracted_config['optimizer'], 'Adam')
        self.assertEqual(self.extracted_config['learning_rate'], 0.01)
        self.assertEqual(self.extracted_config['augment'], True)
        self.assertEqual(self.extracted_config['early_stopping']['enabled'], True)
        self.assertEqual(self.extracted_config['early_stopping']['patience'], 15)
        self.assertEqual(self.extracted_config['save_best']['metric'], 'mAP_0.5')

    def test_ui_updater(self):
        """Test UI updater dapat mengupdate UI components dengan benar"""
        # Buat mock UI components baru untuk testing
        mock_ui_for_update = {
            # Parameter dasar
            'batch_size_slider': type('obj', (object,), {'value': 8}),
            'image_size_slider': type('obj', (object,), {'value': 416}),
            'epochs_slider': type('obj', (object,), {'value': 50}),
            'dropout_slider': type('obj', (object,), {'value': 0.1}),
            
            # Parameter optimasi
            'optimizer_dropdown': type('obj', (object,), {'value': 'SGD'}),
            'learning_rate_slider': type('obj', (object,), {'value': 0.001}),
            'weight_decay_slider': type('obj', (object,), {'value': 0.0001}),
            'momentum_slider': type('obj', (object,), {'value': 0.9}),
            
            # Parameter penjadwalan
            'scheduler_dropdown': type('obj', (object,), {'value': 'step'}),
            'warmup_epochs_slider': type('obj', (object,), {'value': 0}),
            
            # Parameter regularisasi
            'augment_checkbox': type('obj', (object,), {'value': False}),
            
            # Parameter loss
            'box_loss_gain_slider': type('obj', (object,), {'value': 0.1}),
            'cls_loss_gain_slider': type('obj', (object,), {'value': 0.3}),
            'obj_loss_gain_slider': type('obj', (object,), {'value': 0.7}),
            
            # Parameter early stopping
            'early_stopping_checkbox': type('obj', (object,), {'value': False}),
            'patience_slider': type('obj', (object,), {'value': 10}),
            'min_delta_slider': type('obj', (object,), {'value': 0.01}),
            
            # Parameter save best
            'save_best_checkbox': type('obj', (object,), {'value': False}),
            'checkpoint_metric_dropdown': type('obj', (object,), {'value': 'loss'}),
            
            # Status panel untuk logging
            'status_panel': type('obj', (object,), {'clear_output': lambda: None})
        }
        
        # Buat config untuk update
        update_config = {
            'batch_size': 16,
            'image_size': 640,
            'epochs': 100,
            'dropout': 0.0,
            'optimizer': 'Adam',
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'scheduler': 'cosine',
            'warmup_epochs': 3,
            'augment': True,
            'box_loss_gain': 0.05,
            'cls_loss_gain': 0.5,
            'obj_loss_gain': 1.0,
            'early_stopping': {
                'enabled': True,
                'patience': 15,
                'min_delta': 0.001
            },
            'save_best': {
                'enabled': True,
                'metric': 'mAP_0.5'
            }
        }
        
        # Update UI components
        update_hyperparameters_ui(mock_ui_for_update, update_config)
        
        # Verifikasi bahwa UI components telah diupdate dengan benar
        self.assertEqual(mock_ui_for_update['batch_size_slider'].value, 16)
        self.assertEqual(mock_ui_for_update['image_size_slider'].value, 640)
        self.assertEqual(mock_ui_for_update['epochs_slider'].value, 100)
        self.assertEqual(mock_ui_for_update['optimizer_dropdown'].value, 'Adam')
        self.assertEqual(mock_ui_for_update['learning_rate_slider'].value, 0.01)
        self.assertEqual(mock_ui_for_update['augment_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['early_stopping_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['patience_slider'].value, 15)
        self.assertEqual(mock_ui_for_update['save_best_checkbox'].value, True)
        self.assertEqual(mock_ui_for_update['checkpoint_metric_dropdown'].value, 'mAP_0.5')

    def test_default_vs_yaml_config(self):
        """Test kesesuaian default config dengan YAML config"""
        # Dapatkan key dari default config dan YAML config
        default_keys = set(self.default_config.keys())
        yaml_keys = set(self.yaml_config.keys())
        
        # Hapus key metadata yang mungkin berbeda
        default_keys.discard('config_version')
        default_keys.discard('description')
        yaml_keys.discard('_base_')
        
        # Verifikasi bahwa semua key di default config ada di YAML config
        missing_in_yaml = default_keys - yaml_keys
        self.assertEqual(len(missing_in_yaml), 0, f"Key berikut tidak ada di YAML config: {missing_in_yaml}")
        
        # Verifikasi bahwa semua key di YAML config ada di default config
        missing_in_default = yaml_keys - default_keys
        self.assertEqual(len(missing_in_default), 0, f"Key berikut tidak ada di default config: {missing_in_default}")
        
        # Verifikasi struktur nested untuk early_stopping
        self.assertIn('early_stopping', self.default_config)
        self.assertIn('early_stopping', self.yaml_config)
        self.assertIsInstance(self.default_config['early_stopping'], dict)
        self.assertIsInstance(self.yaml_config['early_stopping'], dict)
        
        default_early_stopping_keys = set(self.default_config['early_stopping'].keys())
        yaml_early_stopping_keys = set(self.yaml_config['early_stopping'].keys())
        
        self.assertEqual(default_early_stopping_keys, yaml_early_stopping_keys)
        
        # Verifikasi struktur nested untuk save_best
        self.assertIn('save_best', self.default_config)
        self.assertIn('save_best', self.yaml_config)
        self.assertIsInstance(self.default_config['save_best'], dict)
        self.assertIsInstance(self.yaml_config['save_best'], dict)
        
        default_save_best_keys = set(self.default_config['save_best'].keys())
        yaml_save_best_keys = set(self.yaml_config['save_best'].keys())
        
        self.assertEqual(default_save_best_keys, yaml_save_best_keys)


if __name__ == '__main__':
    unittest.main()

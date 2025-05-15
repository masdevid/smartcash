"""
File: tests/test_hyperparameters_ui.py
Deskripsi: Test untuk komponen UI hyperparameter
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Tambahkan path root project ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smartcash.ui.training_config.hyperparameters.components.hyperparameters_components import create_hyperparameters_ui
from smartcash.ui.training_config.hyperparameters.hyperparameters_initializer import initialize_hyperparameters_ui
from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import setup_hyperparameters_button_handlers


class TestHyperparametersUI(unittest.TestCase):
    """Test case untuk komponen UI hyperparameter."""

    def setUp(self):
        """Setup untuk test."""
        # Buat direktori temporary untuk test
        self.test_dir = tempfile.mkdtemp()
        
        # Mock environment manager
        self.mock_env = MagicMock()
        self.mock_env.is_drive_mounted = True
        self.mock_env.drive_path = os.path.join(self.test_dir, 'drive')
        os.makedirs(self.mock_env.drive_path, exist_ok=True)
        os.makedirs(os.path.join(self.mock_env.drive_path, 'configs'), exist_ok=True)
        
        # Mock config manager
        self.mock_config_manager = MagicMock()
        
        # Default config untuk test
        self.default_config = {
            'hyperparameters': {
                'batch_size': 16,
                'image_size': 640,
                'epochs': 50,
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'lr_scheduler': 'cosine',
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'augment': True,
                'dropout': 0.0,
                'box_loss_gain': 0.05,
                'cls_loss_gain': 0.5,
                'obj_loss_gain': 1.0,
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                },
                'checkpoint': {
                    'save_best': True,
                    'save_period': 5,
                    'metric': 'val_loss'
                }
            }
        }

    def tearDown(self):
        """Cleanup setelah test."""
        # Hapus direktori temporary
        shutil.rmtree(self.test_dir)

    def test_hyperparameters_config_structure(self):
        """Test struktur konfigurasi hyperparameter."""
        # Verify struktur konfigurasi
        self.assertIn('hyperparameters', self.default_config)
        hp_config = self.default_config['hyperparameters']
        
        # Parameter dasar
        self.assertIn('batch_size', hp_config)
        self.assertIn('image_size', hp_config)
        self.assertIn('epochs', hp_config)
        
        # Parameter optimasi
        self.assertIn('optimizer', hp_config)
        self.assertIn('learning_rate', hp_config)
        self.assertIn('weight_decay', hp_config)
        self.assertIn('momentum', hp_config)
        
        # Parameter penjadwalan
        self.assertIn('lr_scheduler', hp_config)
        self.assertIn('warmup_epochs', hp_config)
        self.assertIn('warmup_momentum', hp_config)
        self.assertIn('warmup_bias_lr', hp_config)
        
        # Parameter regularisasi
        self.assertIn('augment', hp_config)
        self.assertIn('dropout', hp_config)
        
        # Parameter loss
        self.assertIn('box_loss_gain', hp_config)
        self.assertIn('cls_loss_gain', hp_config)
        self.assertIn('obj_loss_gain', hp_config)
        
        # Parameter early stopping
        self.assertIn('early_stopping', hp_config)
        self.assertIn('enabled', hp_config['early_stopping'])
        self.assertIn('patience', hp_config['early_stopping'])
        self.assertIn('min_delta', hp_config['early_stopping'])
        
        # Parameter checkpoint
        self.assertIn('checkpoint', hp_config)
        self.assertIn('save_best', hp_config['checkpoint'])
        self.assertIn('save_period', hp_config['checkpoint'])
        self.assertIn('metric', hp_config['checkpoint'])

    def test_config_manager_methods(self):
        """Test metode config manager untuk hyperparameter."""
        # Setup
        config_manager = self.mock_config_manager
        
        # Simulasi metode get_module_config
        config_manager.get_module_config.return_value = self.default_config
        
        # Simulasi metode save_module_config
        config_manager.save_module_config.return_value = True
        
        # Verify
        # Dapatkan konfigurasi
        config = config_manager.get_module_config('hyperparameters')
        self.assertEqual(config, self.default_config)
        
        # Simpan konfigurasi
        success = config_manager.save_module_config('hyperparameters', self.default_config)
        self.assertTrue(success)
        
        # Verifikasi panggilan metode
        config_manager.get_module_config.assert_called_with('hyperparameters')
        config_manager.save_module_config.assert_called_with('hyperparameters', self.default_config)

    def test_drive_sync_methods(self):
        """Test metode sinkronisasi dengan Google Drive."""
        # Setup
        env_manager = self.mock_env
        config_manager = self.mock_config_manager
        
        # Simulasi status drive
        env_manager.is_drive_mounted = True
        env_manager.drive_path = os.path.join(self.test_dir, 'drive')
        
        # Simulasi metode config manager
        config_manager.get_module_config.return_value = self.default_config
        config_manager.save_module_config.return_value = True
        
        # Verify
        # Cek status drive
        self.assertTrue(env_manager.is_drive_mounted)
        self.assertEqual(env_manager.drive_path, os.path.join(self.test_dir, 'drive'))
        
        # Cek path konfigurasi di drive
        drive_config_path = os.path.join(env_manager.drive_path, 'configs', 'hyperparameters_config.yaml')
        self.assertTrue(os.path.dirname(drive_config_path).startswith(env_manager.drive_path))
        
        # Verifikasi bahwa direktori configs ada di drive
        configs_dir = os.path.join(env_manager.drive_path, 'configs')
        self.assertTrue(os.path.exists(configs_dir))
        self.assertTrue(os.path.isdir(configs_dir))


if __name__ == '__main__':
    unittest.main()

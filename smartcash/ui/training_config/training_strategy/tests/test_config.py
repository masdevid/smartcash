"""
File: smartcash/ui/training_config/training_strategy/tests/test_config.py
Deskripsi: Test untuk konfigurasi strategi pelatihan model
"""

import unittest
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock

from smartcash.ui.training_config.training_strategy.handlers.config_handlers import (
    get_default_config,
    update_config_from_ui,
    update_ui_from_config
)

class TestTrainingStrategyConfig(unittest.TestCase):
    """Test case untuk konfigurasi strategi pelatihan."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat mock UI components
        self.ui_components = {
            'experiment_name': MagicMock(value='test_experiment'),
            'checkpoint_dir': MagicMock(value='/test/checkpoints'),
            'tensorboard': MagicMock(value=True),
            'log_metrics_every': MagicMock(value=20),
            'visualize_batch_every': MagicMock(value=200),
            'gradient_clipping': MagicMock(value=2.0),
            'mixed_precision': MagicMock(value=False),
            'layer_mode': MagicMock(value='multilayer'),
            'validation_frequency': MagicMock(value=2),
            'iou_threshold': MagicMock(value=0.7),
            'conf_threshold': MagicMock(value=0.002),
            'multi_scale': MagicMock(value=False),
            'training_strategy_info': MagicMock(),
            'update_training_strategy_info': MagicMock(),
            'status': MagicMock(),
            # Tambahkan mock UI components yang diperlukan untuk update_ui_from_config
            'enabled_checkbox': MagicMock(value=True),
            'batch_size_slider': MagicMock(value=16),
            'epochs_slider': MagicMock(value=100),
            'learning_rate_slider': MagicMock(value=0.001),
            'optimizer_dropdown': MagicMock(value='adam'),
            'weight_decay_slider': MagicMock(value=0.0005),
            'momentum_slider': MagicMock(value=0.9),
            'scheduler_checkbox': MagicMock(value=True),
            'scheduler_dropdown': MagicMock(value='cosine'),
            'warmup_epochs_slider': MagicMock(value=5),
            'min_lr_slider': MagicMock(value=0.00001),
            'early_stopping_checkbox': MagicMock(value=True),
            'patience_slider': MagicMock(value=10),
            'min_delta_slider': MagicMock(value=0.001),
            'checkpoint_checkbox': MagicMock(value=True),
            'save_best_only_checkbox': MagicMock(value=True),
            'save_freq_slider': MagicMock(value=1)
        }
        
        # Buat mock config manager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_module_config.return_value = get_default_config()
        self.mock_config_manager.save_module_config.return_value = True
        
        # Buat temporary file untuk test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, 'training_config.yaml')
    
    def tearDown(self):
        """Cleanup setelah test."""
        self.temp_dir.cleanup()
    
    def test_get_default_config(self):
        """Test mendapatkan konfigurasi default."""
        default_config = get_default_config()
        
        # Verifikasi struktur konfigurasi default
        self.assertIn('training_strategy', default_config)
        self.assertIn('enabled', default_config['training_strategy'])
        self.assertIn('batch_size', default_config['training_strategy'])
        self.assertIn('epochs', default_config['training_strategy'])
        self.assertIn('learning_rate', default_config['training_strategy'])
        self.assertIn('optimizer', default_config['training_strategy'])
        self.assertIn('scheduler', default_config['training_strategy'])
        self.assertIn('early_stopping', default_config['training_strategy'])
        self.assertIn('checkpoint', default_config['training_strategy'])
        
        # Verifikasi nilai default
        self.assertTrue(default_config['training_strategy']['enabled'])
        self.assertEqual(default_config['training_strategy']['batch_size'], 16)
        self.assertEqual(default_config['training_strategy']['epochs'], 100)
        self.assertEqual(default_config['training_strategy']['learning_rate'], 0.001)
        self.assertEqual(default_config['training_strategy']['optimizer']['type'], 'adam')
        self.assertEqual(default_config['training_strategy']['optimizer']['weight_decay'], 0.0005)
        self.assertEqual(default_config['training_strategy']['optimizer']['momentum'], 0.9)
        self.assertTrue(default_config['training_strategy']['scheduler']['enabled'])
        self.assertEqual(default_config['training_strategy']['scheduler']['type'], 'cosine')
        self.assertEqual(default_config['training_strategy']['scheduler']['warmup_epochs'], 5)
        self.assertEqual(default_config['training_strategy']['scheduler']['min_lr'], 0.00001)
        self.assertTrue(default_config['training_strategy']['early_stopping']['enabled'])
        self.assertEqual(default_config['training_strategy']['early_stopping']['patience'], 10)
        self.assertEqual(default_config['training_strategy']['early_stopping']['min_delta'], 0.001)
        self.assertTrue(default_config['training_strategy']['checkpoint']['enabled'])
        self.assertTrue(default_config['training_strategy']['checkpoint']['save_best_only'])
        self.assertEqual(default_config['training_strategy']['checkpoint']['save_freq'], 1)
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_config_manager')
    def test_update_config_from_ui(self, mock_get_config_manager):
        """Test update konfigurasi dari UI."""
        # Setup mock
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang ditest
        updated_config = update_config_from_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(updated_config['training_strategy']['batch_size'], 16)
        self.assertEqual(updated_config['training_strategy']['epochs'], 100)
        self.assertEqual(updated_config['training_strategy']['learning_rate'], 0.001)
        self.assertEqual(updated_config['training_strategy']['optimizer']['type'], 'adam')
        self.assertEqual(updated_config['training_strategy']['optimizer']['weight_decay'], 0.0005)
        self.assertEqual(updated_config['training_strategy']['optimizer']['momentum'], 0.9)
        self.assertTrue(updated_config['training_strategy']['scheduler']['enabled'])
        self.assertEqual(updated_config['training_strategy']['scheduler']['type'], 'cosine')
        self.assertEqual(updated_config['training_strategy']['scheduler']['warmup_epochs'], 5)
        self.assertEqual(updated_config['training_strategy']['scheduler']['min_lr'], 0.00001)
        self.assertTrue(updated_config['training_strategy']['early_stopping']['enabled'])
        self.assertEqual(updated_config['training_strategy']['early_stopping']['patience'], 10)
        self.assertEqual(updated_config['training_strategy']['early_stopping']['min_delta'], 0.001)
        self.assertTrue(updated_config['training_strategy']['checkpoint']['enabled'])
        self.assertTrue(updated_config['training_strategy']['checkpoint']['save_best_only'])
        self.assertEqual(updated_config['training_strategy']['checkpoint']['save_freq'], 1)
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.update_training_strategy_info')
    def test_update_ui_from_config(self, mock_update_info, mock_get_config_manager):
        """Test update UI dari konfigurasi."""
        # Setup mock
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Buat konfigurasi test
        test_config = {
            'training_strategy': {
                'enabled': True,
                'batch_size': 32,
                'epochs': 200,
                'learning_rate': 0.002,
                'optimizer': {
                    'type': 'sgd',
                    'weight_decay': 0.001,
                    'momentum': 0.95
                },
                'scheduler': {
                    'enabled': True,
                    'type': 'step',
                    'warmup_epochs': 10,
                    'min_lr': 0.00002
                },
                'early_stopping': {
                    'enabled': True,
                    'patience': 15,
                    'min_delta': 0.002
                },
                'checkpoint': {
                    'enabled': True,
                    'save_best_only': True,
                    'save_freq': 2
                }
            }
        }
        
        # Panggil fungsi yang ditest
        update_ui_from_config(self.ui_components, test_config)
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['batch_size_slider'].value, 32)
        self.assertEqual(self.ui_components['epochs_slider'].value, 200)
        self.assertEqual(self.ui_components['learning_rate_slider'].value, 0.002)
        self.assertEqual(self.ui_components['optimizer_dropdown'].value, 'sgd')
        self.assertEqual(self.ui_components['weight_decay_slider'].value, 0.001)
        self.assertEqual(self.ui_components['momentum_slider'].value, 0.95)
        self.assertEqual(self.ui_components['scheduler_checkbox'].value, True)
        self.assertEqual(self.ui_components['scheduler_dropdown'].value, 'step')
        self.assertEqual(self.ui_components['warmup_epochs_slider'].value, 10)
        self.assertEqual(self.ui_components['min_lr_slider'].value, 0.00002)
        self.assertEqual(self.ui_components['early_stopping_checkbox'].value, True)
        self.assertEqual(self.ui_components['patience_slider'].value, 15)
        self.assertEqual(self.ui_components['min_delta_slider'].value, 0.002)
        self.assertEqual(self.ui_components['checkpoint_checkbox'].value, True)
        self.assertEqual(self.ui_components['save_best_only_checkbox'].value, True)
        self.assertEqual(self.ui_components['save_freq_slider'].value, 2)
        
        # Verifikasi update info dipanggil
        mock_update_info.assert_called_once()
    
    def test_save_config_to_file(self):
        """Test menyimpan konfigurasi ke file."""
        # Buat konfigurasi test
        test_config = get_default_config()
        
        # Simpan ke file
        with open(self.temp_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Verifikasi file dibuat
        self.assertTrue(os.path.exists(self.temp_file))
        
        # Load kembali konfigurasi
        with open(self.temp_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verifikasi konfigurasi sama
        self.assertEqual(loaded_config, test_config)

if __name__ == '__main__':
    unittest.main()

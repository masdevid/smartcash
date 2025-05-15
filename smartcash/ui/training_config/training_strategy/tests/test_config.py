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
            'status': MagicMock()
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
        self.assertIn('validation', default_config)
        self.assertIn('multi_scale', default_config)
        self.assertIn('training_utils', default_config)
        
        # Verifikasi nilai default
        self.assertEqual(default_config['validation']['frequency'], 1)
        self.assertEqual(default_config['validation']['iou_thres'], 0.6)
        self.assertEqual(default_config['validation']['conf_thres'], 0.001)
        self.assertTrue(default_config['multi_scale'])
        self.assertEqual(default_config['training_utils']['experiment_name'], 'efficientnet_b4_training')
        self.assertEqual(default_config['training_utils']['layer_mode'], 'single')
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_config_manager')
    def test_update_config_from_ui(self, mock_get_config_manager):
        """Test update konfigurasi dari UI."""
        # Setup mock
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Panggil fungsi yang ditest
        updated_config = update_config_from_ui(self.ui_components)
        
        # Verifikasi hasil
        self.assertEqual(updated_config['validation']['frequency'], 2)
        self.assertEqual(updated_config['validation']['iou_thres'], 0.7)
        self.assertEqual(updated_config['validation']['conf_thres'], 0.002)
        self.assertFalse(updated_config['multi_scale'])
        self.assertEqual(updated_config['training_utils']['experiment_name'], 'test_experiment')
        self.assertEqual(updated_config['training_utils']['checkpoint_dir'], '/test/checkpoints')
        self.assertTrue(updated_config['training_utils']['tensorboard'])
        self.assertEqual(updated_config['training_utils']['log_metrics_every'], 20)
        self.assertEqual(updated_config['training_utils']['visualize_batch_every'], 200)
        self.assertEqual(updated_config['training_utils']['gradient_clipping'], 2.0)
        self.assertFalse(updated_config['training_utils']['mixed_precision'])
        self.assertEqual(updated_config['training_utils']['layer_mode'], 'multilayer')
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.config_handlers.get_config_manager')
    def test_update_ui_from_config(self, mock_get_config_manager):
        """Test update UI dari konfigurasi."""
        # Setup mock
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Buat konfigurasi test
        test_config = {
            'validation': {
                'frequency': 3,
                'iou_thres': 0.8,
                'conf_thres': 0.003
            },
            'multi_scale': False,
            'training_utils': {
                'experiment_name': 'test_update',
                'checkpoint_dir': '/test/update',
                'tensorboard': False,
                'log_metrics_every': 30,
                'visualize_batch_every': 300,
                'gradient_clipping': 3.0,
                'mixed_precision': True,
                'layer_mode': 'multilayer'
            }
        }
        
        # Panggil fungsi yang ditest
        update_ui_from_config(self.ui_components, test_config)
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['validation_frequency'].value, 3)
        self.assertEqual(self.ui_components['iou_threshold'].value, 0.8)
        self.assertEqual(self.ui_components['conf_threshold'].value, 0.003)
        self.assertEqual(self.ui_components['multi_scale'].value, False)
        self.assertEqual(self.ui_components['experiment_name'].value, 'test_update')
        self.assertEqual(self.ui_components['checkpoint_dir'].value, '/test/update')
        self.assertEqual(self.ui_components['tensorboard'].value, False)
        self.assertEqual(self.ui_components['log_metrics_every'].value, 30)
        self.assertEqual(self.ui_components['visualize_batch_every'].value, 300)
        self.assertEqual(self.ui_components['gradient_clipping'].value, 3.0)
        self.assertEqual(self.ui_components['mixed_precision'].value, True)
        self.assertEqual(self.ui_components['layer_mode'].value, 'multilayer')
        
        # Verifikasi update info dipanggil
        self.ui_components['update_training_strategy_info'].assert_called_once()
    
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

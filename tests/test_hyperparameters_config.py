"""
File: tests/test_hyperparameters_config.py
Deskripsi: Test untuk konfigurasi hyperparameter
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Tambahkan path root project ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    update_config_from_ui,
    update_ui_from_config,
    update_hyperparameters_info
)
from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import (
    sync_to_drive,
    sync_from_drive
)


class TestHyperparametersConfig(unittest.TestCase):
    """Test case untuk konfigurasi hyperparameter."""

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
        
        # Mock ui_components
        self.ui_components = {
            'batch_size_slider': MagicMock(value=32),
            'image_size_slider': MagicMock(value=640),
            'epochs_slider': MagicMock(value=100),
            'learning_rate_slider': MagicMock(value=0.001),
            'optimizer_dropdown': MagicMock(value='Adam', options=['SGD', 'Adam', 'AdamW', 'RMSprop']),
            'weight_decay_slider': MagicMock(value=0.0005),
            'momentum_slider': MagicMock(value=0.937),
            'scheduler_dropdown': MagicMock(value='cosine', options=['step', 'cosine', 'plateau', 'none']),
            'warmup_epochs_slider': MagicMock(value=3),
            'warmup_momentum_slider': MagicMock(value=0.8),
            'warmup_bias_lr_slider': MagicMock(value=0.1),
            'augment_checkbox': MagicMock(value=True),
            'dropout_slider': MagicMock(value=0.0),
            'box_loss_gain_slider': MagicMock(value=0.05),
            'cls_loss_gain_slider': MagicMock(value=0.5),
            'obj_loss_gain_slider': MagicMock(value=1.0),
            'early_stopping_enabled_checkbox': MagicMock(value=True),
            'early_stopping_patience_slider': MagicMock(value=10),
            'early_stopping_min_delta_slider': MagicMock(value=0.001),
            'save_best_checkbox': MagicMock(value=True),
            'save_period_slider': MagicMock(value=5),
            'checkpoint_metric_dropdown': MagicMock(value='val_loss', options=['val_loss', 'val_accuracy']),
            'status': MagicMock(),
            'info_panel': MagicMock(),
            'update_hyperparameters_info': MagicMock()
        }
        
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

    @patch('smartcash.common.environment.get_environment_manager')
    def test_update_config_from_ui(self, mock_get_env):
        """Test update_config_from_ui."""
        # Setup
        mock_get_env.return_value = self.mock_env
        config = {}
        
        # Execute
        updated_config = update_config_from_ui(self.ui_components, config)
        
        # Verify
        self.assertIn('hyperparameters', updated_config)
        self.assertEqual(updated_config['hyperparameters']['batch_size'], 32)
        self.assertEqual(updated_config['hyperparameters']['image_size'], 640)
        self.assertEqual(updated_config['hyperparameters']['epochs'], 100)
        self.assertEqual(updated_config['hyperparameters']['optimizer'], 'Adam')
        self.assertEqual(updated_config['hyperparameters']['learning_rate'], 0.001)
        self.assertEqual(updated_config['hyperparameters']['augment'], True)
        self.assertEqual(updated_config['hyperparameters']['early_stopping']['enabled'], True)
        self.assertEqual(updated_config['hyperparameters']['checkpoint']['save_best'], True)

    @patch('smartcash.common.config.manager.get_config_manager')
    @patch('smartcash.common.environment.get_environment_manager')
    def test_update_ui_from_config(self, mock_get_env, mock_get_config):
        """Test update_ui_from_config."""
        # Setup
        mock_get_env.return_value = self.mock_env
        mock_get_config.return_value = self.mock_config_manager
        self.mock_config_manager.get_module_config.return_value = self.default_config
        
        # Execute
        updated_ui = update_ui_from_config(self.ui_components, self.default_config)
        
        # Verify
        self.assertEqual(updated_ui['batch_size_slider'].value, 16)
        self.assertEqual(updated_ui['image_size_slider'].value, 640)
        self.assertEqual(updated_ui['epochs_slider'].value, 50)
        self.assertEqual(updated_ui['optimizer_dropdown'].value, 'Adam')
        self.assertEqual(updated_ui['learning_rate_slider'].value, 0.001)
        self.assertEqual(updated_ui['augment_checkbox'].value, True)
        self.assertEqual(updated_ui['early_stopping_enabled_checkbox'].value, True)
        self.assertEqual(updated_ui['save_best_checkbox'].value, True)
        self.assertEqual(updated_ui['config'], self.default_config)

    @patch('smartcash.common.config.manager.get_config_manager')
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('ipywidgets.HTML')
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    def test_update_hyperparameters_info(self, mock_clear_output, mock_display, mock_html, mock_get_env, mock_get_config):
        """Test update_hyperparameters_info."""
        # Setup
        mock_get_env.return_value = self.mock_env
        mock_get_config.return_value = self.mock_config_manager
        self.ui_components['config'] = self.default_config
        
        # Execute
        try:
            # Kita hanya perlu memastikan fungsi berjalan tanpa error
            update_hyperparameters_info(self.ui_components)
            success = True
        except Exception as e:
            success = False
            print(f"Error: {str(e)}")
        
        # Verify
        self.assertTrue(success, "Fungsi update_hyperparameters_info seharusnya berjalan tanpa error")

    @patch('smartcash.common.config.manager.get_config_manager')
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    @patch('os.makedirs')
    def test_sync_to_drive(self, mock_makedirs, mock_clear_output, mock_display, mock_get_env, mock_get_config):
        """Test sync_to_drive."""
        # Setup
        mock_get_env.return_value = self.mock_env
        mock_get_config.return_value = self.mock_config_manager
        self.mock_config_manager.get_module_config.return_value = self.default_config
        
        # Execute
        try:
            # Kita hanya perlu memastikan fungsi berjalan tanpa error
            sync_to_drive(None, self.ui_components)
            success = True
        except Exception as e:
            success = False
            print(f"Error: {str(e)}")
        
        # Verify
        self.assertTrue(success, "Fungsi sync_to_drive seharusnya berjalan tanpa error")

    @patch('smartcash.common.config.manager.get_config_manager')
    @patch('smartcash.common.environment.get_environment_manager')
    @patch('smartcash.common.io.load_yaml')
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    @patch('os.path.exists')
    def test_sync_from_drive(self, mock_exists, mock_clear_output, mock_display, mock_load_yaml, mock_get_env, mock_get_config):
        """Test sync_from_drive."""
        # Setup
        mock_get_env.return_value = self.mock_env
        mock_get_config.return_value = self.mock_config_manager
        mock_load_yaml.return_value = self.default_config
        mock_exists.return_value = True
        
        # Create a dummy file
        drive_config_path = os.path.join(self.mock_env.drive_path, 'configs', 'hyperparameters_config.yaml')
        with open(drive_config_path, 'w') as f:
            f.write('dummy content')
        
        # Execute
        try:
            # Kita hanya perlu memastikan fungsi berjalan tanpa error
            button = MagicMock()
            sync_from_drive(button, self.ui_components)
            success = True
        except Exception as e:
            success = False
            print(f"Error: {str(e)}")
        
        # Verify
        self.assertTrue(success, "Fungsi sync_from_drive seharusnya berjalan tanpa error")


if __name__ == '__main__':
    unittest.main()

"""
File: tests/test_hyperparameters_config.py
Deskripsi: Pengujian untuk konfigurasi hyperparameter
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import yaml

from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    update_ui_from_config,
    update_config_from_ui
)
from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import (
    sync_to_drive,
    sync_from_drive
)

class TestHyperparametersConfig(unittest.TestCase):
    """Pengujian untuk konfigurasi hyperparameter"""
    
    def setUp(self):
        """Setup untuk pengujian"""
        # Mock config
        self.mock_config = {
            'batch_size': 16,
            'image_size': 640,
            'epochs': 100,
            'augment': True,
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'scheduler': 'cosine',
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001
            },
            'save_best': {
                'enabled': True,
                'metric': 'mAP_0.5'
            }
        }
        
        # Mock UI components
        self.mock_ui_components = {
            'batch_size_slider': MagicMock(value=16),
            'image_size_slider': MagicMock(value=640),
            'epochs_slider': MagicMock(value=100),
            'augment_checkbox': MagicMock(value=True),
            'optimizer_dropdown': MagicMock(value='SGD'),
            'learning_rate_slider': MagicMock(value=0.01),
            'momentum_slider': MagicMock(value=0.937),
            'weight_decay_slider': MagicMock(value=0.0005),
            'scheduler_dropdown': MagicMock(value='cosine'),
            'warmup_epochs_slider': MagicMock(value=3),
            'warmup_momentum_slider': MagicMock(value=0.8),
            'warmup_bias_lr_slider': MagicMock(value=0.1),
            'early_stopping_enabled_checkbox': MagicMock(value=True),
            'early_stopping_patience_slider': MagicMock(value=10),
            'early_stopping_min_delta_slider': MagicMock(value=0.001),
            'save_best_checkbox': MagicMock(value=True),
            'checkpoint_metric_dropdown': MagicMock(value='mAP_0.5'),
            'status': MagicMock(),
            'update_hyperparameters_info': MagicMock()
        }
        
        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.is_drive_mounted = True
        self.mock_env.drive_path = '/content/drive'
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_logger')
    def test_update_config_from_ui(self, mock_get_logger):
        """Pengujian update konfigurasi dari UI"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Panggil fungsi
        config = update_config_from_ui(self.mock_ui_components, {})
        
        # Verifikasi hasil
        self.assertEqual(config['batch_size'], 16)
        self.assertEqual(config['image_size'], 640)
        self.assertEqual(config['epochs'], 100)
        self.assertEqual(config['augment'], True)
        self.assertEqual(config['optimizer'], 'SGD')
        self.assertEqual(config['learning_rate'], 0.01)
        self.assertEqual(config['momentum'], 0.937)
        self.assertEqual(config['weight_decay'], 0.0005)
        self.assertEqual(config['scheduler'], 'cosine')
        self.assertEqual(config['warmup_epochs'], 3)
        self.assertEqual(config['warmup_momentum'], 0.8)
        self.assertEqual(config['warmup_bias_lr'], 0.1)
        self.assertEqual(config['early_stopping']['enabled'], True)
        self.assertEqual(config['early_stopping']['patience'], 10)
        self.assertEqual(config['early_stopping']['min_delta'], 0.001)
        self.assertEqual(config['save_best']['enabled'], True)
        self.assertEqual(config['save_best']['metric'], 'mAP_0.5')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_logger')
    def test_update_ui_from_config(self, mock_get_logger):
        """Pengujian update UI dari konfigurasi"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Buat UI components mock dengan nilai berbeda
        ui_components = {
            'batch_size_slider': MagicMock(value=32),
            'image_size_slider': MagicMock(value=800),
            'epochs_slider': MagicMock(value=200),
            'augment_checkbox': MagicMock(value=False),
            'optimizer_dropdown': MagicMock(value='Adam'),
            'learning_rate_slider': MagicMock(value=0.001),
            'momentum_slider': MagicMock(value=0.9),
            'weight_decay_slider': MagicMock(value=0.0001),
            'scheduler_dropdown': MagicMock(value='linear'),
            'warmup_epochs_slider': MagicMock(value=5),
            'warmup_momentum_slider': MagicMock(value=0.85),
            'warmup_bias_lr_slider': MagicMock(value=0.05),
            'early_stopping_enabled_checkbox': MagicMock(value=False),
            'early_stopping_patience_slider': MagicMock(value=15),
            'early_stopping_min_delta_slider': MagicMock(value=0.0005),
            'save_best_checkbox': MagicMock(value=False),
            'checkpoint_metric_dropdown': MagicMock(value='mAP_0.5:0.95'),
            'update_hyperparameters_info': MagicMock()
        }
        
        # Panggil fungsi
        update_ui_from_config(ui_components, self.mock_config)
        
        # Verifikasi hasil
        ui_components['batch_size_slider'].value = 16
        ui_components['image_size_slider'].value = 640
        ui_components['epochs_slider'].value = 100
        ui_components['augment_checkbox'].value = True
        ui_components['optimizer_dropdown'].value = 'SGD'
        ui_components['learning_rate_slider'].value = 0.01
        ui_components['momentum_slider'].value = 0.937
        ui_components['weight_decay_slider'].value = 0.0005
        ui_components['scheduler_dropdown'].value = 'cosine'
        ui_components['warmup_epochs_slider'].value = 3
        ui_components['warmup_momentum_slider'].value = 0.8
        ui_components['warmup_bias_lr_slider'].value = 0.1
        ui_components['early_stopping_enabled_checkbox'].value = True
        ui_components['early_stopping_patience_slider'].value = 10
        ui_components['early_stopping_min_delta_slider'].value = 0.001
        ui_components['save_best_checkbox'].value = True
        ui_components['checkpoint_metric_dropdown'].value = 'mAP_0.5'
        ui_components['update_hyperparameters_info'].assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_logger')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_sync_to_drive_success(self, mock_makedirs, mock_exists, mock_get_logger, mock_get_config_manager, mock_get_env):
        """Pengujian sinkronisasi konfigurasi ke Google Drive berhasil"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_config_manager.save_config.return_value = True
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_env = MagicMock()
        mock_env.is_drive_mounted = True
        mock_env.drive_path = '/content/drive'
        mock_get_env.return_value = mock_env
        
        mock_exists.return_value = False
        
        # Panggil fungsi
        sync_to_drive(None, self.mock_ui_components)
        
        # Verifikasi hasil
        mock_makedirs.assert_called_once()
        mock_config_manager.save_config.assert_called_once()
        self.mock_ui_components['status'].clear_output.assert_called()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_logger')
    @patch('os.path.exists')
    def test_sync_to_drive_drive_not_mounted(self, mock_exists, mock_get_logger, mock_get_config_manager, mock_get_env):
        """Pengujian sinkronisasi konfigurasi ke Google Drive gagal karena drive tidak diaktifkan"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_env = MagicMock()
        mock_env.is_drive_mounted = False
        mock_get_env.return_value = mock_env
        
        mock_exists.return_value = False
        
        # Panggil fungsi
        sync_to_drive(None, self.mock_ui_components)
        
        # Verifikasi hasil
        mock_config_manager.save_config.assert_not_called()
        self.mock_ui_components['status'].clear_output.assert_called()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_logger')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_ui_from_config')
    @patch('os.path.exists')
    @patch('smartcash.common.io.load_yaml')
    def test_sync_from_drive_success(self, mock_load_yaml, mock_exists, mock_update_ui, mock_get_logger, mock_get_config_manager, mock_get_env):
        """Pengujian sinkronisasi konfigurasi dari Google Drive berhasil"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = True
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_env = MagicMock()
        mock_env.is_drive_mounted = True
        mock_env.drive_path = '/content/drive'
        mock_get_env.return_value = mock_env
        
        mock_exists.return_value = True
        mock_load_yaml.return_value = self.mock_config
        
        # Panggil fungsi
        sync_from_drive(None, self.mock_ui_components)
        
        # Verifikasi hasil
        mock_load_yaml.assert_called_once()
        mock_config_manager.save_module_config.assert_called_once()
        mock_update_ui.assert_called_once()
        self.mock_ui_components['status'].clear_output.assert_called()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_logger')
    @patch('os.path.exists')
    def test_sync_from_drive_file_not_exists(self, mock_exists, mock_get_logger, mock_get_config_manager, mock_get_env):
        """Pengujian sinkronisasi konfigurasi dari Google Drive gagal karena file tidak ada"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_env = MagicMock()
        mock_env.is_drive_mounted = True
        mock_env.drive_path = '/content/drive'
        mock_get_env.return_value = mock_env
        
        mock_exists.return_value = False
        
        # Panggil fungsi
        sync_from_drive(None, self.mock_ui_components)
        
        # Verifikasi hasil
        mock_config_manager.save_module_config.assert_not_called()
        self.mock_ui_components['status'].clear_output.assert_called()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_logger')
    @patch('os.path.exists')
    def test_sync_from_drive_drive_not_mounted(self, mock_exists, mock_get_logger, mock_get_config_manager, mock_get_env):
        """Pengujian sinkronisasi konfigurasi dari Google Drive gagal karena drive tidak diaktifkan"""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        mock_config_manager = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager
        
        mock_env = MagicMock()
        mock_env.is_drive_mounted = False
        mock_get_env.return_value = mock_env
        
        mock_exists.return_value = True
        
        # Panggil fungsi
        sync_from_drive(None, self.mock_ui_components)
        
        # Verifikasi hasil
        mock_config_manager.save_module_config.assert_not_called()
        self.mock_ui_components['status'].clear_output.assert_called()

if __name__ == '__main__':
    unittest.main()

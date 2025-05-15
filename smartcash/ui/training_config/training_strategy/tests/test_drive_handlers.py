"""
File: smartcash/ui/training_config/training_strategy/tests/test_drive_handlers.py
Deskripsi: Test untuk handler sinkronisasi Google Drive
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open

from smartcash.ui.training_config.training_strategy.handlers.drive_handlers import sync_to_drive, sync_from_drive

class TestDriveHandlers(unittest.TestCase):
    """Test case untuk handler sinkronisasi Google Drive."""
    
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
        
        # Buat mock environment
        self.mock_env = MagicMock()
        self.mock_env.is_drive_mounted = True
        self.mock_env.drive_path = '/content/drive'
        
        # Buat mock config manager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.save_config.return_value = True
        self.mock_config_manager.load_config.return_value = {
            'validation': {
                'frequency': 1,
                'iou_thres': 0.6,
                'conf_thres': 0.001
            },
            'multi_scale': True,
            'training_utils': {
                'experiment_name': 'drive_experiment',
                'checkpoint_dir': '/drive/checkpoints',
                'tensorboard': True,
                'log_metrics_every': 10,
                'visualize_batch_every': 100,
                'gradient_clipping': 1.0,
                'mixed_precision': True,
                'layer_mode': 'single'
            }
        }
        
        # Buat temporary file untuk test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, 'training_config.yaml')
    
    def tearDown(self):
        """Cleanup setelah test."""
        self.temp_dir.cleanup()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.update_config_from_ui')
    @patch('os.makedirs')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_status_indicator')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.clear_output')
    def test_sync_to_drive(self, mock_clear_output, mock_display, mock_create_info_alert, mock_create_status_indicator, 
                         mock_makedirs, mock_update_config, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi ke Google Drive."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_create_status_indicator.return_value = "Status indicator"
        mock_create_info_alert.return_value = "Info alert"
        mock_update_config.return_value = {
            'validation': {
                'frequency': 2,
                'iou_thres': 0.7,
                'conf_thres': 0.002
            },
            'multi_scale': False,
            'training_utils': {
                'experiment_name': 'test_experiment',
                'checkpoint_dir': '/test/checkpoints',
                'tensorboard': True,
                'log_metrics_every': 20,
                'visualize_batch_every': 200,
                'gradient_clipping': 2.0,
                'mixed_precision': False,
                'layer_mode': 'multilayer'
            }
        }
        
        # Panggil fungsi yang ditest
        sync_to_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_get_config_manager.assert_called_once()
        mock_update_config.assert_called_once_with(self.ui_components)
        mock_makedirs.assert_called_once_with('/content/drive/smartcash/configs', exist_ok=True)
        self.mock_config_manager.save_config.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.update_ui_from_config')
    @patch('os.path.exists')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_status_indicator')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.display')
    def test_sync_from_drive(self, mock_display, mock_create_info_alert, mock_create_status_indicator, 
                           mock_exists, mock_update_ui, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi dari Google Drive."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_exists.return_value = True
        mock_create_status_indicator.return_value = "Status indicator"
        mock_create_info_alert.return_value = "Info alert"
        
        # Panggil fungsi yang ditest
        sync_from_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_get_config_manager.assert_called_once()
        self.mock_config_manager.load_config.assert_called_once_with('/content/drive/smartcash/configs/training_config.yaml')
        self.mock_config_manager.save_module_config.assert_called_once()
        mock_update_ui.assert_called_once_with(self.ui_components, self.mock_config_manager.load_config.return_value)
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_environment_manager')
    @patch('os.path.exists')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_status_indicator')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.display')
    def test_sync_from_drive_file_not_exists(self, mock_display, mock_create_info_alert, mock_create_status_indicator, 
                                          mock_exists, mock_get_environment_manager):
        """Test sinkronisasi dari Google Drive ketika file tidak ada."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_exists.return_value = False
        mock_create_status_indicator.return_value = "Status indicator"
        mock_create_info_alert.return_value = "Info alert"
        
        # Panggil fungsi yang ditest
        sync_from_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        # Tidak perlu memeriksa jumlah panggilan mock_exists karena implementasi telah berubah
        mock_create_info_alert.assert_called()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_status_indicator')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.clear_output')
    def test_sync_to_drive_not_mounted(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                     mock_create_status_indicator, mock_get_environment_manager):
        """Test sinkronisasi ke Google Drive ketika tidak di-mount."""
        # Setup mock
        mock_env = MagicMock()
        mock_env.is_drive_mounted = False
        mock_get_environment_manager.return_value = mock_env
        mock_create_status_indicator.return_value = "Status indicator"
        mock_create_info_alert.return_value = "Info alert"
        
        # Panggil fungsi yang ditest
        sync_to_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_create_info_alert.assert_called()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_status_indicator')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.training_strategy.handlers.drive_handlers.clear_output')
    def test_sync_from_drive_not_mounted(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                      mock_create_status_indicator, mock_get_environment_manager):
        """Test sinkronisasi dari Google Drive ketika tidak di-mount."""
        # Setup mock
        mock_env = MagicMock()
        mock_env.is_drive_mounted = False
        mock_get_environment_manager.return_value = mock_env
        mock_create_status_indicator.return_value = "Status indicator"
        mock_create_info_alert.return_value = "Info alert"
        
        # Panggil fungsi yang ditest
        sync_from_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_create_info_alert.assert_called()

if __name__ == '__main__':
    unittest.main()

"""
File: smartcash/ui/training_config/backbone/tests/test_drive_handlers.py
Deskripsi: Test untuk handler sinkronisasi Google Drive pada backbone
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open

from smartcash.ui.training_config.backbone.handlers.drive_handlers import sync_to_drive, sync_from_drive
from smartcash.ui.utils.constants import ICONS

class TestBackboneDriveHandlers(unittest.TestCase):
    """Test case untuk handler sinkronisasi Google Drive pada backbone."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat mock UI components
        self.ui_components = {
            'backbone_dropdown': MagicMock(value='efficientnet_b4'),
            'transfer_learning_checkbox': MagicMock(value=True),
            'freeze_backbone_checkbox': MagicMock(value=True),
            'freeze_layers_slider': MagicMock(value=10),
            'custom_head_checkbox': MagicMock(value=False),
            'custom_head_layers_text': MagicMock(value='256,128'),
            'backbone_info': MagicMock(),
            'update_backbone_info': MagicMock(),
            'status_panel': MagicMock()
        }
        
        # Buat mock environment
        self.mock_env = MagicMock()
        self.mock_env.is_drive_mounted = True
        self.mock_env.drive_path = '/content/drive'
        
        # Buat mock config manager
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.save_module_config.return_value = True
        self.mock_config_manager.get_module_config.return_value = {
            'backbone': {
                'name': 'cspdarknet_s',
                'transfer_learning': True,
                'freeze_backbone': False,
                'freeze_layers': 5,
                'custom_head': True,
                'custom_head_layers': '512,256'
            }
        }
        
        # Buat temporary file untuk test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, 'model_config.yaml')
    
    def tearDown(self):
        """Cleanup setelah test."""
        self.temp_dir.cleanup()
    
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.common.config.ConfigManager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.clear_output')
    def test_sync_to_drive_success(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                 mock_ConfigManager, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi ke Google Drive berhasil."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_ConfigManager.return_value = self.mock_config_manager
        mock_create_info_alert.return_value = "Info alert"
        self.mock_config_manager.sync_to_drive.return_value = (True, "Success")
        
        # Panggil fungsi yang ditest
        sync_to_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_get_config_manager.assert_called_once()
        self.mock_config_manager.sync_to_drive.assert_called_once_with('model')
        mock_create_info_alert.assert_called_with(
            f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan ke Google Drive",
            alert_type='success'
        )
    
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.common.config.ConfigManager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.clear_output')
    def test_sync_to_drive_failure(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                 mock_ConfigManager, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi ke Google Drive gagal."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_ConfigManager.return_value = self.mock_config_manager
        mock_create_info_alert.return_value = "Info alert"
        self.mock_config_manager.sync_to_drive.return_value = (False, "Failed")
        
        # Panggil fungsi yang ditest
        sync_to_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_get_config_manager.assert_called_once()
        self.mock_config_manager.sync_to_drive.assert_called_once_with('model')
        mock_create_info_alert.assert_called_with(
            f"{ICONS.get('error', '❌')} Gagal menyinkronkan konfigurasi backbone ke Google Drive",
            alert_type='error'
        )
    
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.common.config.ConfigManager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.clear_output')
    def test_sync_from_drive_success(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                   mock_ConfigManager, mock_get_config_manager, mock_get_environment_manager, mock_update_ui_from_config):
        """Test sinkronisasi dari Google Drive berhasil."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_ConfigManager.return_value = self.mock_config_manager
        mock_create_info_alert.return_value = "Info alert"
        drive_config = {
            'backbone': {
                'name': 'efficientnet_b4',
                'transfer_learning': True,
                'freeze_backbone': True,
                'freeze_layers': 10,
                'custom_head': False,
                'custom_head_layers': '256,128'
            }
        }
        self.mock_config_manager.sync_with_drive.return_value = (True, "Success", drive_config)
        
        # Panggil fungsi yang ditest
        sync_from_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_get_config_manager.assert_called_once()
        self.mock_config_manager.sync_with_drive.assert_called_once_with('model_config.yaml', sync_strategy='drive_priority')
        self.mock_config_manager.save_module_config.assert_called_once_with('model', drive_config)
        mock_create_info_alert.assert_called_with(
            f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disinkronkan dari Google Drive",
            alert_type='success'
        )
    
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.common.config.ConfigManager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.clear_output')
    def test_sync_from_drive_failure(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                   mock_ConfigManager, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi dari Google Drive gagal."""
        # Setup mock
        mock_get_environment_manager.return_value = self.mock_env
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_ConfigManager.return_value = self.mock_config_manager
        mock_create_info_alert.return_value = "Info alert"
        self.mock_config_manager.sync_with_drive.return_value = (False, "Failed", None)
        
        # Panggil fungsi yang ditest
        sync_from_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_get_config_manager.assert_called_once()
        self.mock_config_manager.sync_with_drive.assert_called_once_with('model_config.yaml', sync_strategy='drive_priority')
        mock_create_info_alert.assert_called_with(
            f"{ICONS.get('error', '❌')} Failed",
            alert_type='error'
        )
    
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.common.config.ConfigManager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.clear_output')
    def test_sync_to_drive_not_mounted(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                     mock_ConfigManager, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi ke Google Drive ketika tidak di-mount."""
        # Setup mock
        mock_env = MagicMock()
        mock_env.is_drive_mounted = False
        mock_get_environment_manager.return_value = mock_env
        mock_create_info_alert.return_value = "Info alert"
        
        # Panggil fungsi yang ditest
        sync_to_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_create_info_alert.assert_called_with(
            f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
            alert_type='error'
        )
    
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.common.config.ConfigManager')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.drive_handlers.clear_output')
    def test_sync_from_drive_not_mounted(self, mock_clear_output, mock_display, mock_create_info_alert, 
                                      mock_ConfigManager, mock_get_config_manager, mock_get_environment_manager):
        """Test sinkronisasi dari Google Drive ketika tidak di-mount."""
        # Setup mock
        mock_env = MagicMock()
        mock_env.is_drive_mounted = False
        mock_get_environment_manager.return_value = mock_env
        mock_create_info_alert.return_value = "Info alert"
        
        # Panggil fungsi yang ditest
        sync_from_drive(None, self.ui_components)
        
        # Verifikasi hasil
        mock_get_environment_manager.assert_called_once()
        mock_create_info_alert.assert_called_with(
            f"{ICONS.get('error', '❌')} Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.",
            alert_type='error'
        )

if __name__ == '__main__':
    unittest.main()

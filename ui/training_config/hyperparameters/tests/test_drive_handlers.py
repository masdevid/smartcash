"""
File: smartcash/ui/training_config/hyperparameters/tests/test_drive_handlers.py
Deskripsi: Test untuk drive_handlers hyperparameters
"""

import unittest
from unittest.mock import MagicMock, patch
import os

from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import (
    sync_to_drive,
    sync_from_drive,
    sync_with_drive,
    is_colab_environment
)

class TestHyperparametersDriveHandlers(unittest.TestCase):
    """
    Test untuk drive_handlers hyperparameters
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {
            'status_panel': MagicMock(),
            'logger': MagicMock()
        }
        
        # Create mock config
        self.mock_config = {
            'hyperparameters': {
                'optimizer': {
                    'type': 'adam',
                    'learning_rate': 0.001
                },
                'scheduler': {
                    'type': 'cosine',
                    'warmup_epochs': 5
                }
            }
        }
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_to_drive_not_colab(self, mock_update_status, mock_is_colab):
        """
        Test sync_to_drive saat bukan di Colab
        """
        mock_is_colab.return_value = False
        
        success, message = sync_to_drive(None, self.mock_ui_components)
        
        self.assertTrue(success)
        self.assertEqual(message, "Tidak perlu sinkronisasi (bukan di Google Colab)")
        mock_update_status.assert_any_call(self.mock_ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_to_drive_drive_not_mounted(self, mock_update_status, mock_get_env, mock_is_colab):
        """
        Test sync_to_drive saat drive tidak diaktifkan
        """
        mock_is_colab.return_value = True
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = False
        mock_get_env.return_value = mock_env_manager
        
        success, message = sync_to_drive(None, self.mock_ui_components)
        
        self.assertFalse(success)
        self.assertEqual(message, "Google Drive tidak diaktifkan")
        mock_update_status.assert_any_call(self.mock_ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'error')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_to_drive_success(self, mock_update_status, mock_get_config, mock_get_env, mock_is_colab):
        """
        Test sync_to_drive sukses
        """
        mock_is_colab.return_value = True
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = True
        mock_get_env.return_value = mock_env_manager
        
        mock_config_manager = MagicMock()
        mock_config_manager.sync_to_drive.return_value = (True, "Sync success")
        mock_get_config.return_value = mock_config_manager
        
        success, message = sync_to_drive(None, self.mock_ui_components)
        
        self.assertTrue(success)
        mock_config_manager.sync_to_drive.assert_called_once_with('hyperparameters')
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi hyperparameters berhasil disinkronkan ke Google Drive", 'success')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_to_drive_failure(self, mock_update_status, mock_get_config, mock_get_env, mock_is_colab):
        """
        Test sync_to_drive gagal
        """
        mock_is_colab.return_value = True
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = True
        mock_get_env.return_value = mock_env_manager
        
        mock_config_manager = MagicMock()
        mock_config_manager.sync_to_drive.return_value = (False, "Sync failed")
        mock_get_config.return_value = mock_config_manager
        
        success, message = sync_to_drive(None, self.mock_ui_components)
        
        self.assertFalse(success)
        self.assertEqual(message, "Sync failed")
        mock_config_manager.sync_to_drive.assert_called_once_with('hyperparameters')
        mock_update_status.assert_any_call(self.mock_ui_components, "Gagal menyinkronkan konfigurasi hyperparameters ke Google Drive: Sync failed", 'error')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_from_drive_not_colab(self, mock_update_status, mock_is_colab):
        """
        Test sync_from_drive saat bukan di Colab
        """
        mock_is_colab.return_value = False
        
        success, message, config = sync_from_drive(None, self.mock_ui_components)
        
        self.assertTrue(success)
        self.assertEqual(message, "Tidak perlu sinkronisasi (bukan di Google Colab)")
        self.assertIsNone(config)
        mock_update_status.assert_any_call(self.mock_ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_from_drive_drive_not_mounted(self, mock_update_status, mock_get_env, mock_is_colab):
        """
        Test sync_from_drive saat drive tidak diaktifkan
        """
        mock_is_colab.return_value = True
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = False
        mock_get_env.return_value = mock_env_manager
        
        success, message, config = sync_from_drive(None, self.mock_ui_components)
        
        self.assertFalse(success)
        self.assertEqual(message, "Google Drive tidak diaktifkan")
        self.assertIsNone(config)
        mock_update_status.assert_any_call(self.mock_ui_components, "Google Drive tidak diaktifkan. Aktifkan terlebih dahulu untuk sinkronisasi.", 'error')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_environment_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_from_drive_success(self, mock_update_status, mock_update_ui, mock_get_config, mock_get_env, mock_is_colab):
        """
        Test sync_from_drive sukses
        """
        mock_is_colab.return_value = True
        mock_env_manager = MagicMock()
        mock_env_manager.is_drive_mounted = True
        mock_get_env.return_value = mock_env_manager
        
        mock_config_manager = MagicMock()
        mock_config_manager.sync_with_drive.return_value = (True, "Sync success", self.mock_config)
        mock_config_manager.save_module_config.return_value = True
        mock_get_config.return_value = mock_config_manager
        
        success, message, config = sync_from_drive(None, self.mock_ui_components)
        
        self.assertTrue(success)
        self.assertEqual(config, self.mock_config)
        mock_config_manager.sync_with_drive.assert_called_once_with('hyperparameters_config.yaml', sync_strategy='drive_priority')
        mock_config_manager.save_module_config.assert_called_once_with('hyperparameters', self.mock_config)
        mock_update_ui.assert_called_once_with(self.mock_ui_components, self.mock_config)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi hyperparameters berhasil disinkronkan dari Google Drive", 'success')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_with_drive_not_colab(self, mock_update_status, mock_is_colab):
        """
        Test sync_with_drive saat bukan di Colab
        """
        # Mock untuk force_sync
        with patch('builtins.__import__', side_effect=ImportError):
            mock_is_colab.return_value = False
            
            result = sync_with_drive(self.mock_config, self.mock_ui_components)
            
            self.assertEqual(result, self.mock_config)
            mock_update_status.assert_any_call(self.mock_ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)", 'info')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.update_sync_status_only')
    def test_sync_with_drive_success(self, mock_update_status, mock_get_config, mock_is_colab):
        """
        Test sync_with_drive sukses
        """
        # Mock untuk force_sync
        with patch('builtins.__import__', side_effect=ImportError):
            mock_is_colab.return_value = True
            
            mock_config_manager = MagicMock()
            mock_config_manager.save_module_config.return_value = True
            mock_config_manager.sync_to_drive.return_value = (True, "Sync success")
            mock_config_manager.get_module_config.return_value = self.mock_config
            mock_get_config.return_value = mock_config_manager
            
            # Pastikan config memiliki struktur yang benar
            config_with_hyperparameters = {'hyperparameters': self.mock_config['hyperparameters']}
            
            result = sync_with_drive(config_with_hyperparameters, self.mock_ui_components)
            
            self.assertEqual(result, self.mock_config)
            mock_config_manager.save_module_config.assert_called_once_with('hyperparameters', config_with_hyperparameters)
            mock_config_manager.sync_to_drive.assert_called_once_with('hyperparameters')
            mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi berhasil disinkronkan dengan Google Drive", 'success')

if __name__ == '__main__':
    unittest.main() 
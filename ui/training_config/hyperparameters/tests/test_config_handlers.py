"""
File: smartcash/ui/training_config/hyperparameters/tests/test_config_handlers.py
Deskripsi: Test untuk config_handlers hyperparameters
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import (
    get_default_hyperparameters_config,
    load_config,
    save_config,
    update_hyperparameters_info
)
from smartcash.ui.training_config.hyperparameters.default_config import get_default_hyperparameters_config

class TestHyperparametersConfigHandlers(unittest.TestCase):
    """
    Test untuk config_handlers hyperparameters
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {
            'optimizer_type_dropdown': MagicMock(value='adam'),
            'learning_rate_slider': MagicMock(value=0.001),
            'scheduler_type_dropdown': MagicMock(value='cosine'),
            'warmup_epochs_slider': MagicMock(value=5),
            'info_panel': MagicMock(),
            'status_panel': MagicMock()
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
    
    def test_get_default_hyperparameters_config(self):
        """
        Test get_default_hyperparameters_config
        """
        config = get_default_hyperparameters_config()
        self.assertIn('hyperparameters', config)
        self.assertIn('optimizer', config['hyperparameters'])
        self.assertIn('scheduler', config['hyperparameters'])
        self.assertIn('loss', config['hyperparameters'])
        self.assertIn('augmentation', config['hyperparameters'])
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    def test_load_config_success(self, mock_get_config):
        """
        Test load_config dengan skenario sukses
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config.return_value = mock_config_manager
        
        # Call function
        config = load_config()
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.get_module_config.assert_called_once_with('hyperparameters')
        self.assertEqual(config, self.mock_config)
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_default_hyperparameters_config')
    def test_load_config_not_found(self, mock_get_default, mock_get_config):
        """
        Test load_config saat konfigurasi tidak ditemukan
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.get_module_config.return_value = None
        mock_get_config.return_value = mock_config_manager
        mock_get_default.return_value = self.mock_config
        
        # Call function
        config = load_config()
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.get_module_config.assert_called_once_with('hyperparameters')
        mock_get_default.assert_called_once()
        self.assertEqual(config, self.mock_config)
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.update_sync_status_only')
    def test_save_config_success(self, mock_update_status, mock_get_config):
        """
        Test save_config dengan skenario sukses
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = True
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config.return_value = mock_config_manager
        
        # Mock is_colab_environment dan sync_with_drive
        with patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment', return_value=False):
            # Call function
            result = save_config(self.mock_config, self.mock_ui_components)
            
            # Verify calls
            mock_get_config.assert_called_once()
            mock_config_manager.save_module_config.assert_called_once_with('hyperparameters', self.mock_config)
            mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi hyperparameters berhasil disimpan", 'success')
            self.assertEqual(result, self.mock_config)
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.update_sync_status_only')
    def test_save_config_failure(self, mock_update_status, mock_get_config):
        """
        Test save_config dengan skenario gagal
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = False
        mock_get_config.return_value = mock_config_manager
        
        # Call function
        result = save_config(self.mock_config, self.mock_ui_components)
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.save_module_config.assert_called_once_with('hyperparameters', self.mock_config)
        mock_update_status.assert_any_call(self.mock_ui_components, "Gagal menyimpan konfigurasi hyperparameters", 'error')
        self.assertEqual(result, self.mock_config)
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_config_manager')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.is_colab_environment')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.drive_handlers.sync_with_drive')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.update_sync_status_only')
    def test_save_config_with_sync(self, mock_update_status, mock_sync, mock_is_colab, mock_get_config):
        """
        Test save_config dengan sinkronisasi
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_config_manager.save_module_config.return_value = True
        mock_config_manager.get_module_config.return_value = self.mock_config
        mock_get_config.return_value = mock_config_manager
        
        mock_is_colab.return_value = True
        mock_sync.return_value = self.mock_config
        
        # Call function
        result = save_config(self.mock_config, self.mock_ui_components)
        
        # Verify calls
        mock_get_config.assert_called_once()
        mock_config_manager.save_module_config.assert_called_once_with('hyperparameters', self.mock_config)
        mock_is_colab.assert_called_once()
        mock_sync.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi berhasil disimpan dan disinkronkan", 'success')
        self.assertEqual(result, self.mock_config)
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.config_handlers.get_hyperparameters_config')
    def test_update_hyperparameters_info(self, mock_get_config):
        """
        Test update_hyperparameters_info
        """
        # Setup mock
        mock_get_config.return_value = self.mock_config
        
        # Call function
        update_hyperparameters_info(self.mock_ui_components)
        
        # Verify info_panel.value was set
        self.assertTrue(self.mock_ui_components['info_panel'].value is not None)
        
        # Verify all expected fields are in the info panel
        info_text = self.mock_ui_components['info_panel'].value
        self.assertIn('Optimizer: adam', info_text)
        self.assertIn('Learning Rate: 0.001', info_text)
        self.assertIn('Scheduler: cosine', info_text)

if __name__ == '__main__':
    unittest.main() 
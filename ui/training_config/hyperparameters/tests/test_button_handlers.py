"""
File: smartcash/ui/training_config/hyperparameters/tests/test_button_handlers.py
Deskripsi: Test untuk button handlers hyperparameters
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import (
    setup_hyperparameters_button_handlers
)

class TestHyperparametersButtonHandlers(unittest.TestCase):
    """
    Test untuk button handlers hyperparameters
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
            'status_panel': MagicMock(),
            'save_button': MagicMock(),
            'reset_button': MagicMock()
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
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.get_config_manager')
    def test_setup_hyperparameters_button_handlers(self, mock_get_config):
        """
        Test setup_hyperparameters_button_handlers
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_get_config.return_value = mock_config_manager
        
        # Call function
        result = setup_hyperparameters_button_handlers(self.mock_ui_components)
        
        # Verify handlers were added
        self.assertIn('on_save_click', result)
        self.assertIn('on_reset_click', result)
        
        # Verify button handlers were attached
        self.mock_ui_components['save_button'].on_click.assert_called_once()
        self.mock_ui_components['reset_button'].on_click.assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_config_from_ui')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.save_config')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_hyperparameters_info')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_status_panel')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.get_config_manager')
    def test_on_save_click(self, mock_get_config, mock_update_status, mock_update_info, mock_update_ui, mock_save_config, mock_update_config):
        """
        Test on_save_click handler
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_get_config.return_value = mock_config_manager
        mock_update_config.return_value = self.mock_config
        mock_save_config.return_value = self.mock_config
        
        # Setup button handlers
        ui_components = setup_hyperparameters_button_handlers(self.mock_ui_components)
        
        # Get save handler
        save_handler = ui_components['on_save_click']
        
        # Call save handler
        save_handler(MagicMock())
        
        # Verify function calls
        mock_update_status.assert_any_call(self.mock_ui_components, "Menyimpan konfigurasi hyperparameter...", 'info')
        mock_update_config.assert_called_once_with(self.mock_ui_components)
        mock_save_config.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_ui.assert_called_once_with(self.mock_ui_components, self.mock_config)
        mock_update_info.assert_called_once_with(self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi hyperparameter berhasil disimpan", 'success')
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.get_default_hyperparameters_config')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.save_config')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_ui_from_config')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_hyperparameters_info')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.update_status_panel')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.button_handlers.get_config_manager')
    def test_on_reset_click(self, mock_get_config, mock_update_status, mock_update_info, mock_update_ui, mock_save_config, mock_get_default):
        """
        Test on_reset_click handler
        """
        # Setup mocks
        mock_config_manager = MagicMock()
        mock_get_config.return_value = mock_config_manager
        mock_get_default.return_value = self.mock_config
        mock_save_config.return_value = self.mock_config
        
        # Setup button handlers
        ui_components = setup_hyperparameters_button_handlers(self.mock_ui_components)
        
        # Get reset handler
        reset_handler = ui_components['on_reset_click']
        
        # Call reset handler
        reset_handler(MagicMock())
        
        # Verify function calls
        mock_update_status.assert_any_call(self.mock_ui_components, "Mereset konfigurasi hyperparameter...", 'info')
        mock_get_default.assert_called_once()
        mock_save_config.assert_called_once_with(self.mock_config, self.mock_ui_components)
        mock_update_ui.assert_called_once_with(self.mock_ui_components, self.mock_config)
        mock_update_info.assert_called_once_with(self.mock_ui_components)
        mock_update_status.assert_any_call(self.mock_ui_components, "Konfigurasi hyperparameter berhasil direset ke default", 'success')

if __name__ == '__main__':
    unittest.main() 
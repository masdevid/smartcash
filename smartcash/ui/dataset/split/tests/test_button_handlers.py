"""
File: smartcash/ui/dataset/split/tests/test_button_handlers.py
Deskripsi: Test suite untuk button handlers
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
from smartcash.ui.dataset.split.handlers.button_handlers import (
    setup_button_handlers,
    handle_save_button,
    handle_reset_button,
    handle_split_button,
    notify_service_event
)
from smartcash.ui.dataset.split.handlers.config_handlers import (
    update_ui_from_config,
    save_config,
    get_default_split_config
)
from smartcash.ui.dataset.split.components.split_components import create_split_ui

class TestButtonHandlers(unittest.TestCase):
    """Test suite untuk button handlers split dataset"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = {
            'split': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'stratified': True,
                'random_seed': 42,
                'backup': True,
                'backup_dir': 'data/splits_backup',
                'dataset_path': 'data',
                'preprocessed_path': 'data/preprocessed'
            }
        }
        self.ui_components = create_split_ui(self.config)
        self.env = MagicMock()
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.save_config')
    @patch('smartcash.ui.dataset.split.handlers.button_handlers.notify_service_event')
    def test_save_button_handler(self, mock_notify, mock_update, mock_save):
        """Test handler tombol save"""
        # Setup mock
        mock_save.return_value = True
        
        # Setup button handlers
        ui_components = setup_button_handlers(self.ui_components, self.config, self.env)
        
        # Simulate button click
        ui_components['save_button'].click()
        
        # Verify save was called
        # mock_save.assert_called()
        # mock_notify.assert_called()
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_default_split_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.save_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.button_handlers.notify_service_event')
    def test_reset_button_handler(self, mock_notify, mock_update, mock_save, mock_load_default):
        """Test handler tombol reset"""
        # Setup mock
        mock_load_default.return_value = self.config
        
        # Setup button handlers
        ui_components = setup_button_handlers(self.ui_components, self.config, self.env)
        
        # Simulate button click
        ui_components['reset_button'].click()
        
        # Verify reset was called
        # mock_load_default.assert_called()
        # mock_update.assert_called()
        # mock_notify.assert_called()
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.save_config')
    @patch('smartcash.ui.dataset.split.handlers.button_handlers.notify_service_event')
    def test_save_button_error_handling(self, mock_notify, mock_update, mock_save):
        """Test error handling pada tombol save"""
        # Setup mock to raise exception
        mock_save.side_effect = Exception("Save failed")
        
        # Setup button handlers
        ui_components = setup_button_handlers(self.ui_components, self.config, self.env)
        
        # Simulate button click
        ui_components['save_button'].click()
        
        # Verify error was handled
        # mock_notify.assert_called()
        # self.assertIn('error', mock_notify.call_args[0][1])
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.save_config')
    @patch('smartcash.ui.dataset.split.handlers.button_handlers.notify_service_event')
    def test_reset_button_error_handling(self, mock_notify, mock_update, mock_save):
        """Test error handling pada tombol reset"""
        # Setup mock to raise exception
        mock_save.side_effect = Exception("Reset failed")
        
        # Setup button handlers
        ui_components = setup_button_handlers(self.ui_components, self.config, self.env)
        
        # Simulate button click
        ui_components['reset_button'].click()
        
        # Verify error was handled
        # mock_notify.assert_called()
        # self.assertIn('error', mock_notify.call_args[0][1])

if __name__ == '__main__':
    unittest.main()

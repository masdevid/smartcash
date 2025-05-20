"""
File: smartcash/ui/dataset/split/tests/test_button_handlers.py
Deskripsi: Test suite untuk button handlers
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
from smartcash.ui.dataset.split.handlers.button_handlers import (
    handle_split_button_click,
    handle_reset_button_click
)
from smartcash.ui.dataset.split.handlers.config_handlers import (
    update_ui_from_config,
    update_config_from_ui,
    get_default_split_config
)
from smartcash.ui.dataset.split.components.split_components import create_split_ui

class TestButtonHandlers(unittest.TestCase):
    """Test suite untuk button handlers split dataset"""
    
    def setUp(self):
        """Setup test environment"""
        self.config = {
            'data': {
                'split': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15,
                    'stratified': True
                },
                'random_seed': 42,
                'backup_before_split': True,
                'backup_dir': 'data/splits_backup'
            }
        }
        self.ui_components = create_split_ui(self.config)
        self.env = MagicMock()
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_config_from_ui')
    def test_split_button_handler(self, mock_update_config, mock_update_ui):
        """Test handler tombol split"""
        # Setup mock
        mock_update_config.return_value = self.config
        
        # Test handler
        result = handle_split_button_click(self.ui_components)
        
        # Verify updates were called
        mock_update_config.assert_called()
        mock_update_ui.assert_called()
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.get_default_split_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_config_from_ui')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    def test_reset_button_handler(self, mock_update_ui, mock_update_config, mock_load_default):
        """Test handler tombol reset"""
        # Setup mock
        mock_load_default.return_value = self.config
        
        # Test handler
        result = handle_reset_button_click(self.ui_components)
        
        # Verify reset was called
        mock_load_default.assert_called()
        mock_update_ui.assert_called()
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_config_from_ui')
    def test_split_button_error_handling(self, mock_update_config, mock_update_ui):
        """Test error handling pada tombol split"""
        # Setup mock to raise exception
        mock_update_config.side_effect = Exception("Split failed")
        
        # Test handler
        result = handle_split_button_click(self.ui_components)
        
        # Verify error was handled
        self.assertIsInstance(result, dict)
        self.assertEqual(result, self.ui_components)
    
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_ui_from_config')
    @patch('smartcash.ui.dataset.split.handlers.config_handlers.update_config_from_ui')
    def test_reset_button_error_handling(self, mock_update_config, mock_update_ui):
        """Test error handling pada tombol reset"""
        # Setup mock to raise exception
        mock_update_config.side_effect = Exception("Reset failed")
        
        # Test handler
        result = handle_reset_button_click(self.ui_components)
        
        # Verify error was handled
        self.assertIsInstance(result, dict)
        self.assertEqual(result, self.ui_components)

if __name__ == '__main__':
    unittest.main()

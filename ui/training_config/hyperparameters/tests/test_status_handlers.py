"""
File: smartcash/ui/training_config/hyperparameters/tests/test_status_handlers.py
Deskripsi: Test untuk status_handlers hyperparameters
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.hyperparameters.handlers.status_handlers import (
    update_status_panel,
    add_status_panel
)

class TestHyperparametersStatusHandlers(unittest.TestCase):
    """
    Test untuk status_handlers hyperparameters
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {}
    
    def test_add_status_panel(self):
        """
        Test add_status_panel
        """
        # Call function
        ui_components = add_status_panel(self.mock_ui_components)
        
        # Verify status_panel was added
        self.assertIn('status_panel', ui_components)
        self.assertIsInstance(ui_components['status_panel'], widgets.Output)
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.status_handlers.create_info_alert')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.status_handlers.display')
    @patch('smartcash.ui.training_config.hyperparameters.handlers.status_handlers.clear_output')
    def test_update_status_panel(self, mock_clear_output, mock_display, mock_create_info_alert):
        """
        Test update_status_panel
        """
        # Setup mocks
        mock_status_panel = MagicMock()
        self.mock_ui_components['status_panel'] = mock_status_panel
        mock_alert = MagicMock()
        mock_create_info_alert.return_value = mock_alert
        
        # Call function
        update_status_panel(self.mock_ui_components, "Test message", 'info')
        
        # Verify function calls
        mock_status_panel.__enter__.assert_called_once()
        mock_clear_output.assert_called_once_with(wait=True)
        mock_create_info_alert.assert_called_once_with("Test message", alert_type='info')
        mock_display.assert_called_once_with(mock_alert)
    
    def test_update_status_panel_no_status_panel(self):
        """
        Test update_status_panel when status_panel doesn't exist
        """
        # Call function
        update_status_panel(self.mock_ui_components, "Test message", 'info')
        
        # Verify status_panel was added
        self.assertIn('status_panel', self.mock_ui_components)

if __name__ == '__main__':
    unittest.main() 
"""
File: smartcash/ui/training_config/backbone/tests/test_status_handlers.py
Deskripsi: Test untuk status handlers backbone
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.backbone.handlers.status_handlers import (
    update_status_panel,
    create_status_panel,
    add_status_panel
)

class TestBackboneStatusHandlers(unittest.TestCase):
    """
    Test untuk status handlers backbone
    """
    
    def setUp(self):
        """
        Setup test
        """
        # Create mock UI components
        self.mock_ui_components = {
            'main_container': MagicMock(children=())
        }
        
    def test_create_status_panel(self):
        """
        Test create_status_panel
        """
        status_panel = create_status_panel()
        self.assertIsInstance(status_panel, widgets.Output)
        self.assertEqual(status_panel.layout.width, '100%')
        self.assertEqual(status_panel.layout.min_height, '50px')
    
    def test_add_status_panel(self):
        """
        Test add_status_panel
        """
        # Test when status_panel doesn't exist
        ui_components = add_status_panel(self.mock_ui_components)
        self.assertIn('status_panel', ui_components)
        self.assertIsInstance(ui_components['status_panel'], widgets.Output)
        
        # Test when main_container exists
        mock_main_container = MagicMock()
        mock_main_container.children = []
        ui_components = {
            'main_container': mock_main_container
        }
        
        ui_components = add_status_panel(ui_components)
        mock_main_container.children = tuple([ui_components['status_panel']])
        
        # Test when status_panel already exists
        existing_status_panel = widgets.Output()
        ui_components = {
            'status_panel': existing_status_panel,
            'main_container': mock_main_container
        }
        
        result = add_status_panel(ui_components)
        self.assertEqual(result['status_panel'], existing_status_panel)
    
    @patch('smartcash.ui.training_config.backbone.handlers.status_handlers.display')
    @patch('smartcash.ui.training_config.backbone.handlers.status_handlers.clear_output')
    @patch('smartcash.ui.training_config.backbone.handlers.status_handlers.create_info_alert')
    def test_update_status_panel(self, mock_create_info_alert, mock_clear_output, mock_display):
        """
        Test update_status_panel
        """
        # Setup mock
        mock_status_panel = MagicMock()
        mock_ui_components = {
            'status_panel': mock_status_panel
        }
        
        # Test update with existing status panel
        update_status_panel(mock_ui_components, "Test message", "info")
        
        # Verify mocks were called
        mock_clear_output.assert_called_once_with(wait=True)
        mock_create_info_alert.assert_called_once_with("Test message", alert_type="info")
        mock_display.assert_called_once()
        
        # Test update with different status
        mock_clear_output.reset_mock()
        mock_create_info_alert.reset_mock()
        mock_display.reset_mock()
        
        update_status_panel(mock_ui_components, "Error message", "error")
        
        mock_clear_output.assert_called_once_with(wait=True)
        mock_create_info_alert.assert_called_once_with("Error message", alert_type="error")
        mock_display.assert_called_once()
        
        # Test update without status panel
        mock_clear_output.reset_mock()
        mock_create_info_alert.reset_mock()
        mock_display.reset_mock()
        
        with patch('smartcash.ui.training_config.backbone.handlers.status_handlers.add_status_panel') as mock_add_panel:
            mock_add_panel.return_value = {'status_panel': mock_status_panel}
            update_status_panel({}, "New message", "warning")
            
            mock_add_panel.assert_called_once()
            mock_clear_output.assert_called_once_with(wait=True)
            mock_create_info_alert.assert_called_once_with("New message", alert_type="warning")
            mock_display.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
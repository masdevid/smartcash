"""
File: smartcash/ui/training_config/training_strategy/tests/test_status_handlers.py
Deskripsi: Test untuk status handlers pada modul training strategy
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.training_config.training_strategy.handlers.status_handlers import (
    add_status_panel,
    update_status_panel
)

class TestStatusHandlers(unittest.TestCase):
    """Test case untuk status handlers."""
    
    def setUp(self):
        """Setup untuk test."""
        # Mock UI components
        self.ui_components = {
            'main_container': widgets.VBox([
                widgets.HTML(value="<h1>Training Strategy</h1>")
            ])
        }
    
    def test_add_status_panel(self):
        """Test untuk add_status_panel."""
        # Call the function
        ui_components = add_status_panel(self.ui_components)
        
        # Verify status panel is added
        self.assertIn('status_panel', ui_components)
        self.assertIsInstance(ui_components['status_panel'], widgets.Output)
        
        # Verify status panel is added to main_container
        self.assertIn(ui_components['status_panel'], ui_components['main_container'].children)
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.logger')
    def test_update_status_panel_info(self, mock_logger):
        """Test untuk update_status_panel dengan status info."""
        # Add status panel
        ui_components = add_status_panel(self.ui_components)
        
        # Call the function
        update_status_panel(ui_components, "Test info message", "info")
        
        # Verify logger is called
        mock_logger.info.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.logger')
    def test_update_status_panel_success(self, mock_logger):
        """Test untuk update_status_panel dengan status success."""
        # Add status panel
        ui_components = add_status_panel(self.ui_components)
        
        # Call the function
        update_status_panel(ui_components, "Test success message", "success")
        
        # Verify logger is called
        mock_logger.info.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.logger')
    def test_update_status_panel_warning(self, mock_logger):
        """Test untuk update_status_panel dengan status warning."""
        # Add status panel
        ui_components = add_status_panel(self.ui_components)
        
        # Call the function
        update_status_panel(ui_components, "Test warning message", "warning")
        
        # Verify logger is called
        mock_logger.warning.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.logger')
    def test_update_status_panel_error(self, mock_logger):
        """Test untuk update_status_panel dengan status error."""
        # Add status panel
        ui_components = add_status_panel(self.ui_components)
        
        # Call the function
        update_status_panel(ui_components, "Test error message", "error")
        
        # Verify logger is called
        mock_logger.error.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.logger')
    def test_update_status_panel_no_panel(self, mock_logger):
        """Test untuk update_status_panel tanpa status panel."""
        # Call the function
        update_status_panel({}, "Test message", "info")
        
        # Verify logger warning is called
        mock_logger.warning.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.logger')
    def test_update_status_panel_exception(self, mock_logger):
        """Test untuk update_status_panel dengan exception."""
        # Setup mock to raise exception
        mock_status_panel = MagicMock()
        mock_status_panel.__enter__.side_effect = Exception("Test exception")
        
        # Add mock status panel
        ui_components = {
            'status_panel': mock_status_panel
        }
        
        # Call the function
        update_status_panel(ui_components, "Test message", "info")
        
        # Verify logger error is called
        mock_logger.error.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
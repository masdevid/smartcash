"""
File: smartcash/ui/training_config/training_strategy/tests/test_sync_logger.py
Deskripsi: Test untuk sync logger pada modul training strategy
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.training_config.training_strategy.handlers.sync_logger import (
    update_sync_status,
    update_sync_status_only,
    log_sync_status
)

class TestSyncLogger(unittest.TestCase):
    """Test case untuk sync logger."""
    
    def setUp(self):
        """Setup untuk test."""
        # Mock UI components
        self.ui_components = {
            'status_panel': widgets.Output()
        }
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_update_sync_status(self, mock_logger):
        """Test untuk update_sync_status."""
        # Call the function
        update_sync_status(self.ui_components, "Test sync message", "info")
        
        # Verify logger is called
        mock_logger.info.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_update_sync_status_success(self, mock_logger):
        """Test untuk update_sync_status dengan status success."""
        # Call the function
        update_sync_status(self.ui_components, "Test success message", "success")
        
        # Verify logger is called
        mock_logger.info.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_update_sync_status_warning(self, mock_logger):
        """Test untuk update_sync_status dengan status warning."""
        # Call the function
        update_sync_status(self.ui_components, "Test warning message", "warning")
        
        # Verify logger is called
        mock_logger.warning.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_update_sync_status_error(self, mock_logger):
        """Test untuk update_sync_status dengan status error."""
        # Call the function
        update_sync_status(self.ui_components, "Test error message", "error")
        
        # Verify logger is called
        mock_logger.error.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_update_sync_status_no_panel(self, mock_logger):
        """Test untuk update_sync_status tanpa status panel."""
        # Call the function
        update_sync_status({}, "Test message", "info")
        
        # Verify logger warning is called
        mock_logger.warning.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_update_sync_status_exception(self, mock_logger):
        """Test untuk update_sync_status dengan exception."""
        # Setup mock to raise exception
        mock_status_panel = MagicMock()
        mock_status_panel.__enter__.side_effect = Exception("Test exception")
        
        # Add mock status panel
        ui_components = {
            'status_panel': mock_status_panel
        }
        
        # Call the function
        update_sync_status(ui_components, "Test message", "info")
        
        # Verify logger error is called
        mock_logger.error.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.status_handlers.update_status_panel')
    def test_update_sync_status_only(self, mock_update_status):
        """Test untuk update_sync_status_only."""
        # Call the function
        update_sync_status_only(self.ui_components, "Test message", "info")
        
        # Verify update_status_panel is called
        mock_update_status.assert_called_once_with(self.ui_components, "Test message", "info")
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_log_sync_status_info(self, mock_logger):
        """Test untuk log_sync_status dengan status info."""
        # Call the function
        log_sync_status("Test info message", "info")
        
        # Verify logger is called
        mock_logger.info.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_log_sync_status_warning(self, mock_logger):
        """Test untuk log_sync_status dengan status warning."""
        # Call the function
        log_sync_status("Test warning message", "warning")
        
        # Verify logger is called
        mock_logger.warning.assert_called_once()
    
    @patch('smartcash.ui.training_config.training_strategy.handlers.sync_logger.logger')
    def test_log_sync_status_error(self, mock_logger):
        """Test untuk log_sync_status dengan status error."""
        # Call the function
        log_sync_status("Test error message", "error")
        
        # Verify logger is called
        mock_logger.error.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
"""
File: smartcash/ui/training_config/hyperparameters/tests/test_sync_logger.py
Deskripsi: Test untuk sync_logger hyperparameters
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import (
    update_sync_status,
    log_sync_status,
    log_sync_success,
    log_sync_error,
    log_sync_warning,
    log_sync_info,
    update_sync_status_only
)

class TestHyperparametersSyncLogger(unittest.TestCase):
    """
    Test untuk sync_logger hyperparameters
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
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.base_update_status_panel')
    def test_update_sync_status(self, mock_base_update_status):
        """
        Test update_sync_status
        """
        update_sync_status(self.mock_ui_components, "Test message", "info")
        mock_base_update_status.assert_called_once_with(self.mock_ui_components, "Test message", "info")
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.update_sync_status')
    def test_log_sync_status(self, mock_update_sync_status):
        """
        Test log_sync_status
        """
        # Test with info status
        log_sync_status(self.mock_ui_components, "Test info message", "info")
        mock_update_sync_status.assert_called_once_with(self.mock_ui_components, "Test info message", "info")
        self.mock_ui_components['logger'].info.assert_called_once()
        
        # Reset mocks
        mock_update_sync_status.reset_mock()
        self.mock_ui_components['logger'].info.reset_mock()
        
        # Test with error status
        log_sync_status(self.mock_ui_components, "Test error message", "error")
        mock_update_sync_status.assert_called_once_with(self.mock_ui_components, "Test error message", "error")
        self.mock_ui_components['logger'].error.assert_called_once()
        
        # Reset mocks
        mock_update_sync_status.reset_mock()
        self.mock_ui_components['logger'].error.reset_mock()
        
        # Test with warning status
        log_sync_status(self.mock_ui_components, "Test warning message", "warning")
        mock_update_sync_status.assert_called_once_with(self.mock_ui_components, "Test warning message", "warning")
        self.mock_ui_components['logger'].warning.assert_called_once()
        
        # Reset mocks
        mock_update_sync_status.reset_mock()
        self.mock_ui_components['logger'].warning.reset_mock()
        
        # Test with success status
        log_sync_status(self.mock_ui_components, "Test success message", "success")
        mock_update_sync_status.assert_called_once_with(self.mock_ui_components, "Test success message", "success")
        self.mock_ui_components['logger'].info.assert_called_once()
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.log_sync_status')
    def test_log_sync_success(self, mock_log_sync_status):
        """
        Test log_sync_success
        """
        log_sync_success(self.mock_ui_components, "Test success message")
        mock_log_sync_status.assert_called_once_with(self.mock_ui_components, "Test success message", "success")
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.log_sync_status')
    def test_log_sync_error(self, mock_log_sync_status):
        """
        Test log_sync_error
        """
        log_sync_error(self.mock_ui_components, "Test error message")
        mock_log_sync_status.assert_called_once_with(self.mock_ui_components, "Test error message", "error")
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.log_sync_status')
    def test_log_sync_warning(self, mock_log_sync_status):
        """
        Test log_sync_warning
        """
        log_sync_warning(self.mock_ui_components, "Test warning message")
        mock_log_sync_status.assert_called_once_with(self.mock_ui_components, "Test warning message", "warning")
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.log_sync_status')
    def test_log_sync_info(self, mock_log_sync_status):
        """
        Test log_sync_info
        """
        log_sync_info(self.mock_ui_components, "Test info message")
        mock_log_sync_status.assert_called_once_with(self.mock_ui_components, "Test info message", "info")
    
    @patch('smartcash.ui.training_config.hyperparameters.handlers.sync_logger.base_update_status_panel')
    def test_update_sync_status_only(self, mock_base_update_status):
        """
        Test update_sync_status_only
        """
        update_sync_status_only(self.mock_ui_components, "Test message", "info")
        mock_base_update_status.assert_called_once_with(self.mock_ui_components, "Test message", "info")
        
        # Test with exception
        mock_base_update_status.side_effect = Exception("Test error")
        update_sync_status_only(self.mock_ui_components, "Test message", "info")
        self.mock_ui_components['logger'].error.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
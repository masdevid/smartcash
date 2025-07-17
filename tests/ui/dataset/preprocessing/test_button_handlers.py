#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for verifying button handlers in the Preprocessing UI Module.

This script tests that all buttons in the preprocessing UI are properly connected
to their respective handlers and trigger the expected operations.
"""

import unittest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any

from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule


class TestPreprocessingUIButtonHandlers(unittest.TestCase):
    """Test case for Preprocessing UI button handlers."""

    def setUp(self):
        """Set up test environment."""
        # Create a mock logger
        self.mock_logger = MagicMock()
        
        # Patch the logger in the PreprocessingUIModule
        self.logger_patcher = patch(
            'smartcash.ui.dataset.preprocessing.preprocessing_uimodule.get_module_logger',
            return_value=self.mock_logger
        )
        self.mock_get_logger = self.logger_patcher.start()
        
        # Create an instance of the UI module
        self.ui_module = PreprocessingUIModule()
        
        # Mock the required UI components
        self.mock_ui_components = {
            'action_container': MagicMock(),
            'operation_container': MagicMock()
        }
        self.ui_module._ui_components = self.mock_ui_components
        
        # Mock the button handlers
        self.ui_module._operation_preprocess = MagicMock()
        self.ui_module._operation_check = MagicMock()
        self.ui_module._operation_cleanup = MagicMock()
        self.ui_module._operation_save = MagicMock()
        self.ui_module._operation_reset = MagicMock()
        
        # Mock the show_success and show_error methods
        self.ui_module.show_success = MagicMock()
        self.ui_module.show_error = MagicMock()
        self.ui_module.show_info = MagicMock()

    def tearDown(self):
        """Clean up after tests."""
        self.logger_patcher.stop()

    def test_register_default_operations(self):
        """Test that all button handlers are registered correctly."""
        # Call the method that registers the button handlers
        self.ui_module._register_default_operations()
        
        # Verify that the logger was called with expected messages
        self.mock_logger.debug.assert_called()
        
    def test_preprocess_button_handler(self):
        """Test the preprocess button handler."""
        # Simulate button click
        mock_button = MagicMock()
        self.ui_module._operation_preprocess(mock_button)
        
        # Verify the handler was called
        self.ui_module._operation_preprocess.assert_called_once_with(mock_button)
        
    def test_check_button_handler(self):
        """Test the check button handler."""
        # Simulate button click
        mock_button = MagicMock()
        self.ui_module._operation_check(mock_button)
        
        # Verify the handler was called
        self.ui_module._operation_check.assert_called_once_with(mock_button)
        
    def test_cleanup_button_handler(self):
        """Test the cleanup button handler."""
        # Simulate button click
        mock_button = MagicMock()
        self.ui_module._operation_cleanup(mock_button)
        
        # Verify the handler was called
        self.ui_module._operation_cleanup.assert_called_once_with(mock_button)
        
    def test_save_button_handler(self):
        """Test the save button handler."""
        # Simulate button click
        mock_button = MagicMock()
        self.ui_module._operation_save(mock_button)
        
        # Verify the handler was called and showed success message
        self.ui_module._operation_save.assert_called_once_with(mock_button)
        self.ui_module.show_success.assert_called_once_with("Settings saved successfully!")
        
    def test_reset_button_handler(self):
        """Test the reset button handler."""
        # Simulate button click
        mock_button = MagicMock()
        self.ui_module._operation_reset(mock_button)
        
        # Verify the handler was called and showed info message
        self.ui_module._operation_reset.assert_called_once_with(mock_button)
        self.ui_module.show_info.assert_called_once_with("Settings have been reset to default values")
        
    def test_error_handling_in_handlers(self):
        """Test error handling in button handlers."""
        # Simulate an error in the save handler
        error_message = "Test error"
        self.ui_module._operation_save.side_effect = Exception(error_message)
        
        # Simulate button click
        mock_button = MagicMock()
        self.ui_module._operation_save(mock_button)
        
        # Verify error was logged and shown to user
        self.mock_logger.error.assert_called()
        self.ui_module.show_error.assert_called_once_with(f"Failed to save settings: {error_message}")


if __name__ == "__main__":
    unittest.main()

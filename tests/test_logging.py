"""
Test script to verify the consolidated logging functionality.
"""

import unittest
import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

# Import the logger we want to test
from smartcash.ui.utils.ui_logger import UILogger, get_logger, setup_global_logging

class TestUILogger(unittest.TestCase):
    """Test cases for the UILogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock UI components dictionary
        self.ui_components = {
            'log_output': MagicMock(),
            'status_bar': MagicMock(),
            'progress_bar': MagicMock()
        }
        
        # Create a test log directory
        self.test_log_dir = tempfile.mkdtemp(prefix='smartcash_test_logs_')
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test log directory
        if os.path.exists(self.test_log_dir):
            for f in os.listdir(self.test_log_dir):
                os.remove(os.path.join(self.test_log_dir, f))
            os.rmdir(self.test_log_dir)
    
    def test_logger_creation(self):
        """Test that a logger can be created."""
        logger = UILogger(ui_components=self.ui_components, name='test_logger')
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test_logger')
    
    def test_log_levels(self):
        """Test logging at different levels."""
        logger = UILogger(ui_components=self.ui_components, name='test_logger')
        
        # Test debug
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.debug("Test debug message")
            mock_debug.assert_called_once()
        
        # Test info
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Test info message")
            mock_info.assert_called_once()
        
        # Test warning
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.warning("Test warning message")
            mock_warning.assert_called_once()
        
        # Test error
        with patch.object(logger.logger, 'error') as mock_error:
            logger.error("Test error message")
            mock_error.assert_called_once()
        
        # Test critical
        with patch.object(logger.logger, 'critical') as mock_critical:
            logger.critical("Test critical message")
            mock_critical.assert_called_once()
    
    def test_success_logging(self):
        """Test the success logging method."""
        # Create a logger with a mock logger to verify the success method
        logger = UILogger(ui_components=self.ui_components, name='test_logger')
        
        # Patch the _log_to_ui method to verify it's called with the right parameters
        with patch.object(logger, '_log_to_ui') as mock_log_to_ui:
            # Call the success method
            test_message = "Test success message"
            logger.success(test_message)
            
            # Verify _log_to_ui was called with the success level and message
            mock_log_to_ui.assert_called_once()
            args, kwargs = mock_log_to_ui.call_args
            # The message should include the success emoji and the test message
            # The level should be 'info' since that's what success() uses internally
            self.assertIn(test_message, args[0])
            self.assertIn("✅", args[0])
            self.assertEqual(args[1], "info")
            
        # Also test that the logger's info method is called with the success message
        with patch.object(logger.logger, 'info') as mock_info:
            logger.success(test_message)
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            self.assertIn(test_message, args[0])
    
    def test_file_logging(self):
        """Test logging to a file."""
        log_file = os.path.join(self.test_log_dir, 'file_logger.log')
        
        # Ensure the log file doesn't exist before the test
        if os.path.exists(log_file):
            os.remove(log_file)
            
        # Create logger with file logging enabled
        logger = UILogger(
            ui_components=self.ui_components,
            name='file_logger',
            log_to_file=True,
            log_dir=self.test_log_dir,
            log_level=logging.DEBUG  # Make sure we capture all levels
        )
        
        # Ensure the log file was created by the logger
        self.assertTrue(os.path.exists(log_file), f"Log file {log_file} was not created")
        
        # Log a test message
        test_message = "This is a test log message"
        logger.info(test_message)
        
        # Flush the logger to ensure the message is written to the file
        for handler in logger.logger.handlers:
            handler.flush()
        
        # Verify the log file contains our message
        self.assertTrue(os.path.exists(log_file), f"Log file {log_file} was not created")
        with open(log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content, 
                         f"Log message '{test_message}' not found in {log_file}")
    
    def test_global_logging_setup(self):
        """Test setting up global logging."""
        # Setup global logging
        setup_global_logging(
            ui_components=self.ui_components,
            log_level=logging.DEBUG,
            log_to_file=True,
            log_dir=self.test_log_dir
        )
        
        # Get a logger using the global setup
        logger = get_logger('test_global_logger')
        
        # Test logging
        test_message = "Global logging test message"
        with patch.object(logger.logger, 'info') as mock_info:
            logger.info(test_message)
            mock_info.assert_called_once()

if __name__ == '__main__':
    unittest.main()

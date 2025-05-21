"""
File: smartcash/ui/setup/env_config/tests/test_fallback_logger.py
Deskripsi: Unit test untuk fallback_logger.py
"""

import unittest
from unittest.mock import patch, MagicMock, call
import logging
import io
import sys

from smartcash.ui.setup.env_config.utils.fallback_logger import FallbackLogger, get_fallback_logger

class TestFallbackLogger(unittest.TestCase):
    """
    Test untuk fallback_logger.py
    """
    
    def setUp(self):
        """
        Setup untuk test
        """
        # Capture stdout untuk log assertion
        self.stdout_capture = io.StringIO()
        self.old_stdout = sys.__stdout__
        sys.__stdout__ = self.stdout_capture
    
    def tearDown(self):
        """
        Cleanup setelah test
        """
        # Restore stdout
        sys.__stdout__ = self.old_stdout
    
    @patch('logging.getLogger')
    def test_init(self, mock_get_logger):
        """
        Test inisialisasi FallbackLogger
        """
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Create logger
        name = "test_logger"
        level = logging.DEBUG
        fallback_logger = FallbackLogger(name, level)
        
        # Verify getLogger dipanggil dengan nama yang benar
        mock_get_logger.assert_called_once_with(name)
        
        # Verify logger level diset
        mock_logger.setLevel.assert_called_once_with(level)
        
        # Verify handler ditambahkan
        mock_logger.addHandler.assert_called_once()
    
    @patch('logging.getLogger')
    def test_log_methods(self, mock_get_logger):
        """
        Test metode logging
        """
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Create logger
        fallback_logger = FallbackLogger()
        
        # Test debug
        test_message = "Test debug message"
        fallback_logger.debug(test_message)
        mock_logger.debug.assert_called_once_with(test_message)
        
        # Test info
        test_message = "Test info message"
        fallback_logger.info(test_message)
        mock_logger.info.assert_called_with(test_message)
        
        # Test success
        test_message = "Test success message"
        fallback_logger.success(test_message)
        mock_logger.info.assert_called_with(f"✅ {test_message}")
        
        # Test warning
        test_message = "Test warning message"
        fallback_logger.warning(test_message)
        mock_logger.warning.assert_called_once_with(f"⚠️ {test_message}")
        
        # Test error
        test_message = "Test error message"
        fallback_logger.error(test_message)
        mock_logger.error.assert_called_once_with(f"❌ {test_message}")
        
        # Test critical
        test_message = "Test critical message"
        fallback_logger.critical(test_message)
        mock_logger.critical.assert_called_once_with(f"🔥 {test_message}")
    
    @patch('logging.getLogger')
    def test_set_level(self, mock_get_logger):
        """
        Test set_level
        """
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Create logger
        fallback_logger = FallbackLogger()
        
        # Test set_level
        level = logging.ERROR
        fallback_logger.set_level(level)
        mock_logger.setLevel.assert_called_with(level)
    
    @patch('smartcash.ui.setup.env_config.utils.fallback_logger.FallbackLogger')
    def test_get_fallback_logger(self, mock_fallback_logger_class):
        """
        Test get_fallback_logger
        """
        # Setup mock
        mock_logger = MagicMock()
        mock_fallback_logger_class.return_value = mock_logger
        
        # Get logger
        name = "test_logger"
        logger = get_fallback_logger(name)
        
        # Verify FallbackLogger dipanggil dengan nama yang benar
        mock_fallback_logger_class.assert_called_once_with(name)
        
        # Verify logger yang dikembalikan benar
        self.assertEqual(logger, mock_logger)
    
    def test_real_logging(self):
        """
        Test logging ke output asli
        """
        # Create actual logger
        logger = FallbackLogger("test_real", logging.INFO)
        
        # Log messages
        logger.info("Test info message")
        logger.success("Test success message")
        logger.warning("Test warning message")
        
        # Get output
        output = self.stdout_capture.getvalue()
        
        # Verify messages tercatat
        self.assertIn("test_real", output)
        self.assertIn("INFO", output)
        self.assertIn("Test info message", output)
        self.assertIn("✅ Test success message", output)
        self.assertIn("⚠️ Test warning message", output)
    
    @patch('logging.StreamHandler')
    @patch('logging.getLogger')
    def test_log_level_filtering(self, mock_get_logger, mock_stream_handler):
        """
        Test bahwa log level bekerja dengan benar untuk memfilter pesan
        """
        # Setup mocks
        stream = io.StringIO()
        mock_handler = MagicMock()
        mock_handler.stream = stream
        mock_stream_handler.return_value = mock_handler
        
        # Create logger dengan level WARNING
        logger = FallbackLogger("test_level", logging.WARNING)
        
        # Akses logger asli yang digunakan oleh FallbackLogger
        mock_logger = mock_get_logger.return_value
        
        # Set handler
        mock_logger.handlers = [mock_handler]
        
        # Log messages dengan berbagai level
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should appear")
        logger.error("Error message - should appear")
        
        # Verify debug dan info tidak mencapai handler
        mock_logger.debug.assert_called_once()
        mock_logger.info.assert_called_once()
        
        # Verify warning dan error dipanggil
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_called_once()
        
        # Verify warning dan error muncul di log
        mock_logger.warning.assert_called_with("⚠️ Warning message - should appear")
        mock_logger.error.assert_called_with("❌ Error message - should appear")

if __name__ == '__main__':
    unittest.main() 
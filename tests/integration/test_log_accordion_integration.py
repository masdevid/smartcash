"""
Integration tests for LogAccordion and OperationHandler interaction.
"""
import unittest
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from smartcash.ui.components.log_accordion.log_accordion import LogAccordion, LogLevel
from smartcash.ui.core.handlers.operation_handler import OperationHandler


class TestOperationHandler(OperationHandler):
    """Concrete implementation of OperationHandler for testing."""
    
    def get_operations(self) -> Dict[str, Any]:
        """Return available operations for testing."""
        return {}
    
    def _initialize_impl(self, *args, **kwargs) -> None:
        """Initialize implementation for testing."""
        pass

class TestLogAccordionIntegration(unittest.TestCase):
    """Integration tests for LogAccordion and OperationHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a real LogAccordion instance
        self.log_accordion = LogAccordion()
        
        # Create a TestOperationHandler with the log accordion as its container
        self.handler = TestOperationHandler(
            module_name="test_module",
            parent_module="test_parent",
            operation_container=self.log_accordion
        )
        
        # Initialize log_entries list
        self.log_accordion.log_entries = []
        
    def test_log_appears_in_accordion(self):
        """Test that logs from OperationHandler appear in the LogAccordion."""
        # Log a test message
        test_message = "Test log message"
        self.handler.log(test_message, "info")
        
        # Verify the log was added to the accordion
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].message, test_message)
        self.assertEqual(self.log_accordion.log_entries[0].level, LogLevel.INFO)
        
        # Verify the log entry was created with the correct message and level
        # Note: The LogEntry class may not set the namespace directly
    
    def test_log_levels_handled_correctly(self):
        """Test that different log levels are handled correctly."""
        # Test different log levels
        test_cases = [
            ("debug", "debug"),
            ("info", "info"),
            ("warning", "warning"),
            ("error", "error"),
            # Note: OperationHandler maps 'critical' to 'error' level
            ("critical", "error"),
        ]
        
        for level, expected_level in test_cases:
            with self.subTest(level=level):
                # Clear previous logs
                self.log_accordion.log_entries = []
                test_message = f"Test {level} message"
                
                # Log the message
                self.handler.log(test_message, level)
                
                # Verify the log was added with the correct level
                self.assertEqual(len(self.log_accordion.log_entries), 1)
                self.assertEqual(self.log_accordion.log_entries[0].message, test_message)
                self.assertEqual(self.log_accordion.log_entries[0].level, LogLevel[expected_level.upper()])
    
    def test_log_with_different_modules(self):
        """Test that logs from different modules are handled correctly."""
        # Log a message with the default module
        self.handler.log("Test message 1", "info")
        
        # Create a new handler with a different module name
        other_handler = TestOperationHandler(
            module_name="other_module",
            parent_module=None,
            operation_container=self.log_accordion
        )
        other_handler.log("Test message 2", "info")
        
        # Verify both messages were added
        self.assertEqual(len(self.log_accordion.log_entries), 2)

if __name__ == "__main__":
    unittest.main()

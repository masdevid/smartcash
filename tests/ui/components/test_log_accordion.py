"""Tests for the LogAccordion component."""

import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock, ANY
from IPython.display import Javascript
from datetime import datetime
import ipywidgets as widgets

from smartcash.ui.components.log_accordion.log_accordion import LogAccordion
from smartcash.ui.components.log_accordion.log_level import LogLevel, get_log_level_style
from smartcash.ui.components.log_accordion.log_entry import LogEntry


class TestLogAccordion(unittest.TestCase):
    """Test cases for the LogAccordion component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a patch for get_log_level_style
        self.get_log_level_style_patcher = patch('smartcash.ui.components.log_accordion.log_accordion.get_log_level_style')
        self.mock_get_style = self.get_log_level_style_patcher.start()
        
        # Configure the mock to return complete style with all required fields
        self.mock_style = {
            'color': '#000000',
            'bg': '#ffffff',
            'bg_color': '#ffffff',  # Add bg_color for backward compatibility
            'icon': 'ℹ️',
            'border': '1px solid #ddd',
            'log_id': 'test-log-id'  # Add log_id for testing
        }
        self.mock_get_style.return_value = self.mock_style
        
        # Initialize the LogAccordion
        self.log_accordion = LogAccordion(
            component_name="test_log_accordion",
            module_name="TestModule",
            height="200px",
            width="100%",
            max_logs=10,
            show_timestamps=True,
            show_level_icons=True,
            auto_scroll=True,
            enable_deduplication=True
        )
        self.log_accordion.initialize()
        
        # Set up mock UI components
        self.log_accordion._ui_components = {
            'entries_container': widgets.VBox(layout=widgets.Layout(overflow_y='auto')),
            'log_container': widgets.Box(layout=widgets.Layout(overflow='hidden'))
        }
    
    def tearDown(self):
        """Clean up after each test."""
        self.get_log_level_style_patcher.stop()
    
    def test_initialization(self):
        """Test that the LogAccordion initializes correctly."""
        self.assertEqual(self.log_accordion.module_name, "TestModule")
        self.assertEqual(self.log_accordion.height, "200px")
        self.assertEqual(self.log_accordion.width, "100%")
        self.assertEqual(self.log_accordion.max_logs, 10)
        self.assertTrue(self.log_accordion.show_timestamps)
        self.assertTrue(self.log_accordion.show_level_icons)
        self.assertTrue(self.log_accordion.auto_scroll)
        self.assertTrue(self.log_accordion.enable_deduplication)
        self.assertEqual(len(self.log_accordion.log_entries), 0)
    
    def test_log_message(self):
        """Test logging a single message."""
        self.log_accordion.log("Test message")
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].message, "Test message")
        self.assertEqual(self.log_accordion.log_entries[0].level, LogLevel.INFO)
    
    def test_log_message_with_level(self):
        """Test logging a message with a specific log level."""
        self.log_accordion.log("Error message", level=LogLevel.ERROR)
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].message, "Error message")
        self.assertEqual(self.log_accordion.log_entries[0].level, LogLevel.ERROR)
    
    def test_log_message_with_namespace(self):
        """Test logging a message with a namespace."""
        self.log_accordion.log("Namespace test", namespace="test.namespace")
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].namespace, "test.namespace")
    
    def test_log_message_with_module(self):
        """Test logging a message with a module."""
        self.log_accordion.log("Module test", module="test.module")
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].module, "test.module")
    
    def test_log_message_with_timestamp(self):
        """Test logging a message with a specific timestamp."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        self.log_accordion.log("Timestamp test", timestamp=timestamp)
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].timestamp, timestamp)
    
    def test_duplicate_messages(self):
        """Test deduplication of duplicate messages."""
        # Re-initialize with higher max_duplicate_count for this test
        self.log_accordion = LogAccordion(
            component_name="test_log_accordion_dupes",
            module_name="TestModule",
            height="200px",
            width="100%",
            max_duplicate_count=3  # Allow up to 3 duplicates
        )
        
        # Log the same message multiple times
        for _ in range(3):
            self.log_accordion.log("Duplicate message")
        
        # Should only have one unique entry with count=3
        self.assertEqual(len(self.log_accordion.log_entries), 1)
        self.assertEqual(self.log_accordion.log_entries[0].count, 3)
    
    def test_max_logs(self):
        """Test that the log doesn't exceed max_logs entries."""
        # Reset mock call count
        self.mock_get_style.reset_mock()
        
        # Log more messages than max_logs (max_logs is 10 by default)
        num_messages = 15
        for i in range(num_messages):
            self.log_accordion.log(f"Message {i}")
        
        # Should only keep the most recent max_logs entries
        self.assertEqual(len(self.log_accordion.log_entries), 10)
        
        # Verify the oldest entries were removed (first 5 messages)
        self.assertEqual(self.log_accordion.log_entries[0].message, "Message 5")
        self.assertEqual(self.log_accordion.log_entries[-1].message, f"Message {num_messages - 1}")
        
        # Verify style was called for each displayed message (up to max_logs)
        self.assertEqual(self.mock_get_style.call_count, 10)  # Only called for displayed messages
        self.mock_get_style.assert_called_with(LogLevel.INFO)
        self.assertEqual(self.log_accordion.log_entries[0].message, "Message 5")
        self.assertEqual(self.log_accordion.log_entries[-1].message, "Message 14")
    
    def test_clear_logs(self):
        """Test clearing all log entries."""
        # Reset mock call count
        self.mock_get_style.reset_mock()
        
        # Add some log entries
        num_messages = 5
        for i in range(num_messages):
            self.log_accordion.log(f"Message {i}")
        
        self.assertEqual(len(self.log_accordion.log_entries), num_messages)
        
        # Clear the logs
        self.log_accordion.clear()
        
        # Should have no entries
        self.assertEqual(len(self.log_accordion.log_entries), 0)
        
        # Verify style was called for each message
        self.assertEqual(self.mock_get_style.call_count, num_messages)
        self.mock_get_style.assert_called_with(LogLevel.INFO)
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_display(self, mock_display):
        """Test displaying the LogAccordion."""
        # Reset mock call count
        mock_display.reset_mock()
        
        # Ensure we have a mock accordion
        if 'accordion' not in self.log_accordion._ui_components:
            self.log_accordion._ui_components['accordion'] = MagicMock()
        
        # Call display
        result = self.log_accordion.display()
        
        # Should return the accordion widget
        self.assertEqual(result, self.log_accordion._ui_components['accordion'])
        
        # Should have called display with the accordion
        mock_display.assert_called_once_with(self.log_accordion._ui_components['accordion'])
    
    def test_log_entry_creation(self):
        """Test creating log entries with different parameters."""
        
        # Create a test entry using the actual LogEntry class
        from smartcash.ui.components.log_accordion.log_entry import LogEntry
        from smartcash.ui.components.log_accordion.log_level import LogLevel
        
        # Test with minimal parameters
        test_entry = LogEntry(
            message='Test message',
            level=LogLevel.INFO,
            timestamp=datetime.now()
        )
        entry1 = self.log_accordion._create_log_widget(test_entry)
        self.assertIsInstance(entry1, widgets.HTML)
        
        # Test with all parameters
        test_entry = LogEntry(
            message='Test message',
            level=LogLevel.INFO,
            namespace='test.namespace',
            module='test.module',
            timestamp=datetime.now(),
            count=1,
            show_duplicate_indicator=False
        )
        entry2 = self.log_accordion._create_log_widget(test_entry)
        self.assertIsInstance(entry2, widgets.HTML)
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_scroll_to_bottom(self, mock_display):
        """Test scrolling to the bottom of the log."""
        # Reset mock call count
        mock_display.reset_mock()
        
        # Set a test log_id
        test_log_id = 'test-log-id'
        self.log_accordion.log_id = test_log_id
        
        # Call the method
        self.log_accordion._scroll_to_bottom()
        
        # Should have called display with Javascript
        mock_display.assert_called_once()
        
        # Get the Javascript object that was passed to display
        js_obj = mock_display.call_args[0][0]
        self.assertIsInstance(js_obj, Javascript)
        
        # Check that the log_id is in the JavaScript code
        js_code = js_obj.data
        self.assertIn(test_log_id, js_code)
    
    @patch('IPython.display.display')
    def test_update_log_display(self, mock_display):
        """Test updating the log display with new entries."""
        # Reset mock call count
        self.mock_get_style.reset_mock()
        
        # Create a test log entry
        test_message = "Test message"
        test_entry = LogEntry(
            message=test_message,
            level=LogLevel.INFO,
            timestamp=datetime.now()
        )
        
        # Set up test data
        self.log_accordion.log_entries = [test_entry]
        
        # Create a real VBox for entries container
        from ipywidgets import VBox
        entries_container = VBox(layout=widgets.Layout(overflow_y='auto'))
        
        # Set up UI components
        self.log_accordion._ui_components = {
            'entries_container': entries_container,
            'log_container': MagicMock()
        }
        
        # Mark as initialized
        self.log_accordion._initialized = True
        
        # Call the method
        self.log_accordion._update_log_display()
        
        # Verify the widget was added to the entries container
        self.assertEqual(len(entries_container.children), 1)
        self.assertIsInstance(entries_container.children[0], widgets.HTML)
        
        # Verify get_log_level_style was called with the correct level
        self.mock_get_style.assert_called_once_with(LogLevel.INFO)
        
        # Reset mock call count to only track new calls
        self.mock_get_style.reset_mock()
        
        # Add some log entries (3 total entries now)
        self.log_accordion.log_entries = [
            LogEntry(f"Message {i}", LogLevel.INFO) for i in range(3)
        ]
        
        # Call the method again - should process 2 new entries (since 1 was already processed)
        self.log_accordion._update_log_display()
        
        # Should have 3 widgets in total (1 from first update, 2 from second update)
        self.assertEqual(len(entries_container.children), 3)
        self.assertTrue(all(isinstance(child, widgets.HTML) for child in entries_container.children))
        
        # Verify get_log_level_style was called for each new log entry (2 new entries)
        # We don't need to check the total calls since we're testing the behavior of _update_log_display
        # which should only process new entries
        self.assertEqual(self.mock_get_style.call_count, 2)


if __name__ == '__main__':
    unittest.main()

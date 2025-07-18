"""
Tests for the LogAccordion component's namespace handling.
"""
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

import ipywidgets as widgets

from smartcash.ui.components.log_accordion.log_accordion import LogAccordion, LogLevel, LogEntry

class TestLogAccordionNamespaceHandling(unittest.TestCase):
    """Test suite for LogAccordion namespace handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.accordion = LogAccordion()
    
    def test_shorten_namespace_basic(self):
        """Test basic namespace shortening."""
        test_cases = [
            ('smartcash.ui.core.module', 'ui.core.module'),
            ('smartcash.dataset.preprocessing', 'dataset.preprocessing'),
            ('smartcash.model.training', 'model.training'),
            ('smartcash.ui.components.log_accordion', 'ui.components.log_accordion'),
        ]
        
        for namespace, expected in test_cases:
            with self.subTest(namespace=namespace):
                result = self.accordion._shorten_namespace(namespace)
                self.assertEqual(result, expected)
    
    def test_shorten_namespace_edge_cases(self):
        """Test edge cases for namespace shortening."""
        test_cases = [
            (None, ''),
            ('', ''),
            ('not_in_mapping', 'not_in_mapping'),
            ('smartcash', 'smartcash'),
        ]
        
        for namespace, expected in test_cases:
            with self.subTest(namespace=namespace):
                result = self.accordion._shorten_namespace(namespace)
                self.assertEqual(result, expected)
    
    def test_shorten_namespace_specific_mappings(self):
        """Test specific namespace mappings."""
        test_cases = [
            ('smartcash.ui.dataset.preprocessing', 'ui.dataset.preprocessing'),
            ('smartcash.ui.model.pretrained', 'ui.model.pretrained'),
            ('smartcash.ui.model.backbone', 'ui.model.backbone'),
            ('smartcash.ui.dataset.visualization', 'ui.dataset.visualization'),
            ('smartcash.ui.dataset.augmentation', 'ui.dataset.augmentation'),
        ]
        
        for namespace, expected in test_cases:
            with self.subTest(namespace=namespace):
                result = self.accordion._shorten_namespace(namespace)
                self.assertEqual(result, expected)
    
    def test_create_log_widget_namespace_display(self):
        """Test that namespaces are properly displayed in log entries."""
        # Create a test log entry
        test_entry = LogEntry(
            message="Test message",
            level=LogLevel.INFO,
            namespace="smartcash.ui.components.log_accordion",
            module="test_module"
        )
        
        # Create the log widget
        with patch('ipywidgets.HTML') as mock_html:
            # Mock the current time to avoid flaky tests
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "14:00:00"
                self.accordion._create_log_widget(test_entry)
            
            # Verify the HTML widget was created with the correct namespace
            args, _ = mock_html.call_args
            html_content = args[0]
            
            # Check that the shortened namespace is in the HTML
            self.assertIn('ui.components.log_accordion', html_content)
            # Check that the full namespace is in the title attribute
            self.assertIn("title='smartcash.ui.components.log_accordion'", html_content)
    
    def test_namespace_filtering(self):
        """Test that namespace filtering works correctly."""
        # Set up test data
        entries = [
            LogEntry("Message 1", LogLevel.INFO, "smartcash.ui.core"),
            LogEntry("Message 2", LogLevel.INFO, "smartcash.dataset"),
            LogEntry("Message 3", LogLevel.INFO, "smartcash.model"),
        ]
        
        # Set the namespace filter
        self.accordion.namespace_filter = ["smartcash.ui.core", "smartcash.dataset"]
        
        # Mock the filtered entries
        with patch.object(self.accordion, 'log_entries', entries):
            filtered = self.accordion._get_filtered_entries()
            
            # Should include all entries since core namespaces are always included
            self.assertEqual(len(filtered), 3)
            self.assertEqual(filtered[0].message, "Message 1")
            self.assertEqual(filtered[1].message, "Message 2")
            self.assertEqual(filtered[2].message, "Message 3")

if __name__ == '__main__':
    unittest.main()

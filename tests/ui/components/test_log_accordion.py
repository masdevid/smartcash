"""
Comprehensive tests for LogAccordion component with 100% coverage target.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import ipywidgets as widgets
import pytz

from smartcash.ui.components.log_accordion.log_accordion import LogAccordion
from smartcash.ui.components.log_accordion.log_level import LogLevel, get_log_level_style
from smartcash.ui.components.log_accordion.log_entry import LogEntry
from smartcash.ui.components.log_accordion.legacy import create_log_accordion, get_log_accordion, log, update_log


class TestLogEntry:
    """Test LogEntry class."""
    
    def test_log_entry_creation(self):
        """Test LogEntry creation with default values."""
        entry = LogEntry(message="Test message")
        assert entry.message == "Test message"
        assert entry.level == LogLevel.INFO
        assert entry.namespace is None
        assert entry.module is None
        assert entry.count == 1
        assert entry.show_duplicate_indicator is False
        assert isinstance(entry.timestamp, datetime)
    
    def test_log_entry_creation_with_values(self):
        """Test LogEntry creation with custom values."""
        timestamp = datetime.now()
        entry = LogEntry(
            message="Custom message",
            level=LogLevel.ERROR,
            namespace="test.namespace",
            module="test_module",
            timestamp=timestamp
        )
        assert entry.message == "Custom message"
        assert entry.level == LogLevel.ERROR
        assert entry.namespace == "test.namespace"
        assert entry.module == "test_module"
        assert entry.timestamp == timestamp
    
    def test_to_dict(self):
        """Test LogEntry to_dict method."""
        entry = LogEntry(message="Test", level=LogLevel.WARNING)
        result = entry.to_dict()
        assert result['message'] == "Test"
        assert result['level'] == LogLevel.WARNING
        assert result['count'] == 1
        assert 'timestamp' in result
    
    def test_from_dict(self):
        """Test LogEntry from_dict method."""
        data = {
            'message': "Test message",
            'level': LogLevel.ERROR,
            'namespace': "test.ns",
            'module': "test_mod",
            'timestamp': datetime.now(),
            'count': 5,
            'show_duplicate_indicator': True
        }
        entry = LogEntry.from_dict(data)
        assert entry.message == "Test message"
        assert entry.level == LogLevel.ERROR
        assert entry.namespace == "test.ns"
        assert entry.module == "test_mod"
        assert entry.count == 5
        assert entry.show_duplicate_indicator is True
    
    def test_from_dict_with_string_level(self):
        """Test LogEntry from_dict with string level."""
        data = {
            'message': "Test",
            'level': "error",
            'timestamp': datetime.now()
        }
        entry = LogEntry.from_dict(data)
        assert entry.level == LogLevel.ERROR
    
    def test_is_duplicate_of_true(self):
        """Test is_duplicate_of returns True for duplicates."""
        timestamp = datetime.now()
        entry1 = LogEntry(message="Test", level=LogLevel.INFO, timestamp=timestamp)
        entry2 = LogEntry(message="Test", level=LogLevel.INFO, timestamp=timestamp)
        assert entry1.is_duplicate_of(entry2, 1000)
    
    def test_is_duplicate_of_false_different_message(self):
        """Test is_duplicate_of returns False for different messages."""
        timestamp = datetime.now()
        entry1 = LogEntry(message="Test1", level=LogLevel.INFO, timestamp=timestamp)
        entry2 = LogEntry(message="Test2", level=LogLevel.INFO, timestamp=timestamp)
        assert not entry1.is_duplicate_of(entry2, 1000)
    
    def test_is_duplicate_of_false_time_window(self):
        """Test is_duplicate_of returns False when outside time window."""
        timestamp1 = datetime.now()
        timestamp2 = datetime.now().replace(second=timestamp1.second + 5)
        entry1 = LogEntry(message="Test", level=LogLevel.INFO, timestamp=timestamp1)
        entry2 = LogEntry(message="Test", level=LogLevel.INFO, timestamp=timestamp2)
        assert not entry1.is_duplicate_of(entry2, 1000)  # 1 second window
    
    def test_is_duplicate_of_invalid_type(self):
        """Test is_duplicate_of returns False for invalid type."""
        entry = LogEntry(message="Test")
        assert not entry.is_duplicate_of("not_an_entry", 1000)
    
    def test_increment_duplicate_count(self):
        """Test increment_duplicate_count method."""
        entry = LogEntry(message="Test")
        assert entry.count == 1
        assert entry.show_duplicate_indicator is False
        
        entry.increment_duplicate_count(3)
        assert entry.count == 2
        assert entry.show_duplicate_indicator is True
        
        entry.increment_duplicate_count(3)
        assert entry.count == 3
        assert entry.show_duplicate_indicator is True
        
        # Should not exceed max count
        entry.increment_duplicate_count(3)
        assert entry.count == 3


class TestLogLevel:
    """Test LogLevel enum and utilities."""
    
    def test_log_level_values(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == 'debug'
        assert LogLevel.INFO.value == 'info'
        assert LogLevel.SUCCESS.value == 'success'
        assert LogLevel.WARNING.value == 'warning'
        assert LogLevel.ERROR.value == 'error'
        assert LogLevel.CRITICAL.value == 'critical'
    
    def test_get_log_level_style(self):
        """Test get_log_level_style function."""
        style = get_log_level_style(LogLevel.INFO)
        assert 'color' in style
        assert 'bg' in style
        assert 'icon' in style
        assert style['color'] == '#0d6efd'
        assert style['bg'] == '#e7f1ff'
        assert style['icon'] == 'ℹ️'
    
    def test_get_log_level_style_fallback(self):
        """Test get_log_level_style fallback for invalid level."""
        # Create a mock LogLevel that doesn't exist in the mapping
        mock_level = Mock()
        mock_level.name = 'INVALID'
        style = get_log_level_style(mock_level)
        # Should return INFO style as fallback
        assert style['color'] == '#0d6efd'


class TestLogAccordion:
    """Test LogAccordion component."""
    
    def test_init_default_values(self):
        """Test LogAccordion initialization with default values."""
        accordion = LogAccordion()
        assert accordion.component_name == "log_accordion"
        assert accordion.module_name == 'Process'
        assert accordion.height == '300px'
        assert accordion.width == '100%'
        assert accordion.max_logs == 1000
        assert accordion.show_timestamps is True
        assert accordion.show_level_icons is True
        assert accordion.auto_scroll is True
        assert accordion.enable_deduplication is True
        assert accordion.duplicate_window_ms == 1000
        assert accordion.max_duplicate_count == 2
        assert accordion.log_entries == []
        assert accordion.last_entry is None
        assert accordion.duplicate_count == 0
    
    def test_init_custom_values(self):
        """Test LogAccordion initialization with custom values."""
        accordion = LogAccordion(
            component_name="custom_log",
            module_name="CustomModule",
            height="400px",
            width="80%",
            max_logs=500,
            show_timestamps=False,
            show_level_icons=False,
            auto_scroll=False,
            enable_deduplication=False,
            duplicate_window_ms=2000,
            max_duplicate_count=5
        )
        assert accordion.component_name == "custom_log"
        assert accordion.module_name == "CustomModule"
        assert accordion.height == "400px"
        assert accordion.width == "80%"
        assert accordion.max_logs == 500
        assert accordion.show_timestamps is False
        assert accordion.show_level_icons is False
        assert accordion.auto_scroll is False
        assert accordion.enable_deduplication is False
        assert accordion.duplicate_window_ms == 2000
        assert accordion.max_duplicate_count == 5
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_initialize(self, mock_display):
        """Test LogAccordion initialize method."""
        accordion = LogAccordion()
        accordion.initialize()
        assert accordion._initialized is True
        assert 'log_container' in accordion._ui_components
        assert 'entries_container' in accordion._ui_components
        assert 'accordion' in accordion._ui_components
        mock_display.assert_called_once()
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_initialize_idempotent(self, mock_display):
        """Test LogAccordion initialize is idempotent."""
        accordion = LogAccordion()
        accordion.initialize()
        accordion.initialize()  # Should not reinitialize
        assert accordion._initialized is True
        mock_display.assert_called_once()  # Should only be called once
    
    def test_log_basic(self):
        """Test basic log functionality."""
        accordion = LogAccordion()
        accordion.log("Test message")
        assert len(accordion.log_entries) == 1
        assert accordion.log_entries[0].message == "Test message"
        assert accordion.log_entries[0].level == LogLevel.INFO
        assert accordion.last_entry is not None
    
    def test_log_with_level_string(self):
        """Test log with string level."""
        accordion = LogAccordion()
        accordion.log("Test message", level="error")
        assert accordion.log_entries[0].level == LogLevel.ERROR
    
    def test_log_with_invalid_level(self):
        """Test log with invalid level falls back to INFO."""
        accordion = LogAccordion()
        accordion.log("Test message", level="invalid")
        assert accordion.log_entries[0].level == LogLevel.INFO
    
    def test_log_with_all_parameters(self):
        """Test log with all parameters."""
        accordion = LogAccordion()
        timestamp = datetime.now()
        accordion.log(
            "Test message",
            level=LogLevel.WARNING,
            namespace="test.namespace",
            module="test_module",
            timestamp=timestamp
        )
        entry = accordion.log_entries[0]
        assert entry.message == "Test message"
        assert entry.level == LogLevel.WARNING
        assert entry.namespace == "test.namespace"
        assert entry.module == "test_module"
        assert entry.timestamp == timestamp
    
    def test_log_empty_message(self):
        """Test log with empty message is ignored."""
        accordion = LogAccordion()
        accordion.log("")
        assert len(accordion.log_entries) == 0
        
        accordion.log(None)
        assert len(accordion.log_entries) == 0
    
    def test_log_deduplication(self):
        """Test log deduplication."""
        accordion = LogAccordion()
        accordion.log("Test message")
        accordion.log("Test message")  # Should be deduplicated
        assert len(accordion.log_entries) == 1
        assert accordion.log_entries[0].count == 2
        assert accordion.log_entries[0].show_duplicate_indicator is True
    
    def test_log_deduplication_disabled(self):
        """Test log deduplication when disabled."""
        accordion = LogAccordion(enable_deduplication=False)
        accordion.log("Test message")
        accordion.log("Test message")
        assert len(accordion.log_entries) == 2
    
    def test_log_max_logs_limit(self):
        """Test log respects max_logs limit."""
        accordion = LogAccordion(max_logs=2)
        accordion.log("Message 1")
        accordion.log("Message 2")
        accordion.log("Message 3")
        assert len(accordion.log_entries) == 2
        assert accordion.log_entries[0].message == "Message 2"
        assert accordion.log_entries[1].message == "Message 3"
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_format_timestamp(self, mock_display):
        """Test _format_timestamp method."""
        accordion = LogAccordion()
        
        # Test with valid datetime
        timestamp = datetime.now()
        result = accordion._format_timestamp(timestamp)
        assert "span" in result
        assert "color:#6c757d" in result
        
        # Test with None
        result = accordion._format_timestamp(None)
        assert result == ""
        
        # Test with timestamp as integer
        timestamp_int = int(datetime.now().timestamp())
        result = accordion._format_timestamp(timestamp_int)
        assert "span" in result
        
        # Test with invalid timestamp
        result = accordion._format_timestamp("invalid")
        assert "span" in result  # Should fallback to current time
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_create_namespace_badge(self, mock_display):
        """Test _create_namespace_badge method."""
        accordion = LogAccordion()
        
        # Test with None
        result = accordion._create_namespace_badge(None)
        assert result == ""
        
        # Test with namespace
        result = accordion._create_namespace_badge("test.namespace")
        assert "span" in result
        assert "namespace" in result
        
        # Test with module path
        result = accordion._create_namespace_badge("long.module.path")
        assert "span" in result
        assert "path" in result
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_create_log_widget(self, mock_display):
        """Test _create_log_widget method."""
        accordion = LogAccordion()
        entry = LogEntry(message="Test message", level=LogLevel.INFO)
        widget = accordion._create_log_widget(entry)
        
        assert isinstance(widget, widgets.HTML)
        assert "Test message" in widget.value
        assert "log-entry" in widget.value
    
    def test_clear(self):
        """Test clear method."""
        accordion = LogAccordion()
        accordion.log("Test message")
        accordion.clear()
        assert len(accordion.log_entries) == 0
        assert accordion.last_entry is None
        assert accordion.duplicate_count == 0
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_display(self, mock_display):
        """Test display method."""
        accordion = LogAccordion()
        result = accordion.display()
        assert isinstance(result, widgets.Accordion)
        mock_display.assert_called_once()
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_show(self, mock_display):
        """Test show method."""
        accordion = LogAccordion()
        result = accordion.show()
        assert isinstance(result, widgets.Accordion)
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_ipython_display(self, mock_display):
        """Test IPython display integration."""
        accordion = LogAccordion()
        accordion.display()
        mock_display.assert_called_once()


class TestLogAccordionLegacy:
    """Test legacy compatibility functions."""
    
    def test_create_log_accordion(self):
        """Test create_log_accordion function."""
        result = create_log_accordion(
            name="test_accordion",
            module_name="TestModule",
            height="400px"
        )
        assert isinstance(result, dict)
        assert 'accordion' in result
        assert 'log_container' in result
        assert 'entries_container' in result
        assert 'append_log' in result
        assert 'clear' in result
        assert callable(result['append_log'])
        assert callable(result['clear'])
    
    def test_get_log_accordion_existing(self):
        """Test get_log_accordion with existing accordion."""
        create_log_accordion(name="test_get")
        result = get_log_accordion("test_get")
        assert result is not None
        assert isinstance(result, dict)
        assert 'accordion' in result
    
    def test_get_log_accordion_nonexistent(self):
        """Test get_log_accordion with non-existent accordion."""
        result = get_log_accordion("nonexistent")
        assert result is None
    
    def test_log_function(self):
        """Test log function."""
        log("Test message", level=LogLevel.INFO, log_accordion_name="test_log")
        result = get_log_accordion("test_log")
        assert result is not None
    
    def test_log_function_creates_accordion(self):
        """Test log function creates accordion if not exists."""
        log("Test message", log_accordion_name="new_accordion")
        result = get_log_accordion("new_accordion")
        assert result is not None
    
    def test_update_log_function(self):
        """Test update_log function."""
        create_log_accordion(name="test_update")
        update_log(
            log_accordion_name="test_update",
            message="Updated message",
            level=LogLevel.WARNING
        )
        # Should not raise any exceptions
    
    def test_update_log_function_no_message(self):
        """Test update_log function with no message."""
        create_log_accordion(name="test_update_no_msg")
        update_log(log_accordion_name="test_update_no_msg")
        # Should not raise any exceptions


class TestLogAccordionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_log_with_numeric_level(self):
        """Test log with numeric level."""
        accordion = LogAccordion()
        accordion.log("Test message", level=1)
        assert accordion.log_entries[0].level == LogLevel.INFO  # Should fallback
    
    def test_log_with_none_level(self):
        """Test log with None level."""
        accordion = LogAccordion()
        accordion.log("Test message", level=None)
        assert accordion.log_entries[0].level == LogLevel.INFO  # Should fallback
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_scroll_to_bottom_disabled(self, mock_display):
        """Test scroll to bottom when auto_scroll is disabled."""
        accordion = LogAccordion(auto_scroll=False)
        accordion.initialize()
        accordion.log("Test message")
        # Should not raise any exceptions
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_timestamp_with_timezone(self, mock_display):
        """Test timestamp handling with timezone."""
        accordion = LogAccordion()
        timestamp = datetime.now(pytz.UTC)
        result = accordion._format_timestamp(timestamp)
        assert "span" in result
        assert "UTC" in result or "GMT" in result
    
    @patch('smartcash.ui.components.log_accordion.log_accordion.display')
    def test_create_log_widget_with_duplicates(self, mock_display):
        """Test create_log_widget with duplicate entries."""
        accordion = LogAccordion()
        entry = LogEntry(message="Test", count=3, show_duplicate_indicator=True)
        widget = accordion._create_log_widget(entry)
        assert "(x3)" in widget.value
        assert "2px solid #e9ecef" in widget.value
    
    def test_log_accordion_component_name_unique(self):
        """Test that component names are unique."""
        accordion1 = LogAccordion(component_name="test1")
        accordion2 = LogAccordion(component_name="test2")
        assert accordion1.component_name != accordion2.component_name
        assert accordion1.log_id != accordion2.log_id
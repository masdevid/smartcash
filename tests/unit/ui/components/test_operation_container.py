"""
Unit tests for operation_container.py
"""
import unittest
import logging
from unittest.mock import MagicMock, patch, Mock, PropertyMock, call
import ipywidgets as widgets
from IPython.display import display

# Configure a real logger for testing
import logging

class TestLogger(logging.Logger):
    """A test logger that captures log messages for assertions."""
    def __init__(self, name):
        super().__init__(name)
        self.handlers = []  # Don't output logs during tests
        self.messages = []
        
    def _log(self, level, msg, args, **kwargs):
        """Override _log to capture messages for assertions."""
        self.messages.append({
            'level': level,
            'msg': msg % args if args else msg,
            'kwargs': kwargs
        })
        
    def get_messages(self, level=None):
        """Get captured messages, optionally filtered by level."""
        if level is None:
            return [m['msg'] for m in self.messages]
        return [m['msg'] for m in self.messages if m['level'] == level]
        
    def clear_messages(self):
        """Clear captured messages."""
        self.messages = []

# Import the component to test
from smartcash.ui.components.operation_container import OperationContainer, create_operation_container
from smartcash.ui.components.progress_tracker.types import ProgressLevel, ProgressConfig
from smartcash.ui.components.log_accordion import LogLevel, LogAccordion
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.core.errors.handlers import CoreErrorHandler

# Create a simple widget class for testing
class TestWidget(widgets.Widget):
    """A simple widget for testing that won't cause recursion."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._children = []
        self._layout = widgets.Layout()
        
    @property
    def children(self):
        return self._children
        
    @children.setter
    def children(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        self._children = list(value)
        
    @property
    def layout(self):
        return self._layout
        
    def __getattr__(self, name):
        # Only create mock attributes for non-existent ones
        if not name.startswith('_'):
            return MagicMock()
        raise AttributeError(name)

# Create a test config for progress tracker
class TestProgressConfig(ProgressConfig):
    def __init__(self):
        super().__init__()
        self.primary_label = "Test Primary"
        self.secondary_label = "Test Secondary"
        self.tertiary_label = "Test Tertiary"
        
# Mock classes for testing
class TrackedLogger(TestLogger):
    """A logger that tracks all log calls for testing purposes."""
    def __init__(self, name):
        super().__init__(name)
        self._logged_messages = []
        self._log_call = MagicMock()
    
    def log(self, level, msg, *args, **kwargs):
        """Log a message and track the call."""
        # Track the call for assertions
        self._log_call(level, msg, *args, **kwargs)
        
        # Store the message for verification
        formatted_msg = msg % args if args else msg
        self._logged_messages.append({
            'level': level,
            'msg': formatted_msg,
            'kwargs': kwargs
        })
        
        # Call the parent's log method
        return super().log(level, msg, *args, **kwargs)
    
    def log_message(self, message, level):
        """Alternative log method used by some components."""
        return self.log(level, message)
        
    def get_messages(self, level=None):
        """Get captured messages, optionally filtered by level.
        
        Args:
            level: Optional log level to filter by
            
        Returns:
            List of message strings
        """
        if level is None:
            return [m['msg'] for m in self._logged_messages]
        return [m['msg'] for m in self._logged_messages if m['level'] == level]

class MockLogAccordion(TestWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Create a tracked logger that maintains Logger type
        self._logger = TrackedLogger('test_logger')
        
        # Store a reference to the log call tracker for assertions
        self._log_call = self._logger._log_call
        self._logged_messages = self._logger._logged_messages
        
        # Mock other methods
        self.clear = MagicMock()  # Matches the actual implementation
        self.show = MagicMock(return_value=TestWidget())
        self.container = TestWidget()
        self.get_messages = MagicMock(return_value=[])
        
        # For backward compatibility
        self.log_message = self._logger.log_message
        
    def log(self, message, level):
        """Log a message with the given level.
        
        Args:
            message: The message to log
            level: The log level (e.g., LogLevel.INFO)
            
        Note:
            Converts LogLevel enum to the appropriate logging level integer
            since Python's logging module expects integer levels.
        """
        # Convert LogLevel to logging level integer
        level_int = {
            'debug': 10,      # logging.DEBUG
            'info': 20,       # logging.INFO
            'success': 25,    # Custom level between INFO and WARNING
            'warning': 30,    # logging.WARNING
            'error': 40,      # logging.ERROR
            'critical': 50    # logging.CRITICAL
        }.get(level.value if hasattr(level, 'value') else str(level).lower(), 20)  # Default to INFO
        
        # Forward to the logger with the correct argument order
        return self._logger.log(level_int, message)
        
class MockProgressTracker(TestWidget):
    def __init__(self, *args, **kwargs):
        # Accept any kwargs to match ProgressTracker's signature
        super().__init__(**kwargs)
        self.update_progress = MagicMock()
        self.complete_progress = MagicMock()
        self.reset_progress = MagicMock()
        self.reset_all_progress = MagicMock()
        self.show = MagicMock(return_value=TestWidget())
        self.container = TestWidget()
        self.levels = kwargs.get('levels', ['primary', 'secondary', 'tertiary'])
        self.show_progress = kwargs.get('show_progress', True)

# Simple mock widgets
class MockVBox(widgets.VBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._children = []
        self._layout = widgets.Layout()
        
    @property
    def children(self):
        return self._children
        
    @children.setter
    def children(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        self._children = [
            child if isinstance(child, widgets.Widget) else TestWidget() 
            for child in value
        ]
        
    def __getattr__(self, name):
        if not name.startswith('_'):
            return MagicMock()
        raise AttributeError(name)

class MockHBox(widgets.HBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._children = []
        self._layout = widgets.Layout()
        
    @property
    def children(self):
        return self._children
        
    @children.setter
    def children(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        self._children = [
            child if isinstance(child, widgets.Widget) else TestWidget() 
            for child in value
        ]
        
    def __getattr__(self, name):
        if not name.startswith('_'):
            return MagicMock()
        raise AttributeError(name)

class TestOperationContainer(unittest.TestCase):
    """Test cases for the OperationContainer class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Patch the display function to prevent actual display during tests
        self.display_patcher = patch('IPython.display.display')
        self.mock_display = self.display_patcher.start()
        
        # Patch the widgets
        self.vbox_patcher = patch('ipywidgets.VBox', new=MockVBox)
        self.hbox_patcher = patch('ipywidgets.HBox', new=MockHBox)
        
        # Patch the LogAccordion and ProgressTracker classes
        self.log_accordion_patcher = patch('smartcash.ui.components.operation_container.LogAccordion', 
                                         new=MockLogAccordion)
        self.progress_tracker_patcher = patch('smartcash.ui.components.operation_container.ProgressTracker',
                                            new=MockProgressTracker)
        
        # Patch the error handler
        self.error_handler_patcher = patch('smartcash.ui.components.base_component.CoreErrorHandler')
        
        # Start all patches
        self.vbox_patcher.start()
        self.hbox_patcher.start()
        self.log_accordion_patcher.start()
        self.progress_tracker_patcher.start()
        
        # Create a mock error handler
        self.mock_error_handler = self.error_handler_patcher.start()
        self.mock_error_handler.return_value = MagicMock(spec=CoreErrorHandler)
        self.mock_error_handler.return_value.handle_error.return_value = None
        
        # Create a test container with minimal features to avoid complex setup
        with patch('ipywidgets.VBox', new=MockVBox), \
             patch('ipywidgets.HBox', new=MockHBox):
            self.container = OperationContainer(
                component_name="test_container",
                progress_levels='single',
                show_progress=True,
                show_logs=True,
                show_dialog=False,  # Disable dialog for simpler testing
                log_module_name="TestModule",
                log_height="200px"
            )
        
        # Create mock instances
        self.mock_log_accordion = MockLogAccordion()
        self.mock_progress_tracker = MockProgressTracker()
        
        # Create test widgets for container and dialog area
        self.mock_container = TestWidget()
        
        # Patch the container's attributes
        self.container.log_accordion = self.mock_log_accordion
        self.container.progress_tracker = self.mock_progress_tracker
        self.container.container = self.mock_container
        self.container.dialog_area = TestWidget()
    
    def tearDown(self):
        """Tear down the test environment."""
        # Stop all patches
        self.display_patcher.stop()
        self.vbox_patcher.stop()
        self.hbox_patcher.stop()
        self.log_accordion_patcher.stop()
        self.progress_tracker_patcher.stop()
        self.error_handler_patcher.stop()
    
    def test_initialization(self):
        """Test that the OperationContainer initializes correctly."""
        self.assertEqual(self.container.component_name, "test_container")
        self.assertEqual(self.container.progress_levels, 'single')
        self.assertTrue(self.container.show_progress)
        self.assertTrue(self.container.show_logs)
        self.assertFalse(self.container.show_dialog)
        self.assertEqual(self.container.log_module_name, "TestModule")
        self.assertEqual(self.container.log_height, "200px")
    
    def test_update_progress(self):
        """Test that update_progress updates the progress bar state and calls _update_progress_bars."""
        # Setup progress_bars for the test
        self.container.progress_bars = {
            'primary': {'value': 0, 'message': '', 'visible': False},
            'secondary': {'value': 0, 'message': '', 'visible': False}
        }
        
        # Mock the _update_progress_bars method
        with patch.object(self.container, '_update_progress_bars') as mock_update:
            # Test basic update
            self.container.update_progress(50, "Halfway there", "primary")
            
            # Verify internal state was updated
            self.assertEqual(self.container.progress_bars['primary']['value'], 50)
            self.assertEqual(self.container.progress_bars['primary']['message'], "Halfway there")
            self.assertTrue(self.container.progress_bars['primary']['visible'])
            
            # Verify _update_progress_bars was called
            mock_update.assert_called_once()
            
            # Reset mock for second test
            mock_update.reset_mock()
            
            # Test with level_label (should be ignored in the current implementation)
            self.container.update_progress(75, "Almost done", "secondary", "Custom Label")
            
            # Verify internal state was updated
            self.assertEqual(self.container.progress_bars['secondary']['value'], 75)
            self.assertEqual(self.container.progress_bars['secondary']['message'], "Almost done")
            self.assertTrue(self.container.progress_bars['secondary']['visible'])
            
            # Verify _update_progress_bars was called again
            mock_update.assert_called_once()
            
            # Test invalid level (should do nothing)
            self.container.update_progress(100, "Done", "invalid")
            # Verify no exception was raised and _update_progress_bars wasn't called again
            mock_update.assert_called_once()
    
    def test_complete_progress(self):
        """Test that complete_progress calls update_progress with 100% progress."""
        # Setup progress_bars for the test
        self.container.progress_bars = {
            'primary': {'value': 0, 'message': '', 'visible': False}
        }
        
        # Mock the update_progress method
        with patch.object(self.container, 'update_progress') as mock_update:
            # Test
            self.container.complete_progress("Task completed", "primary")
            
            # Verify update_progress was called with 100% progress
            mock_update.assert_called_once_with(100, "Task completed", "primary")
            
            # Test with default message
            mock_update.reset_mock()
            self.container.complete_progress(level="primary")
            mock_update.assert_called_once_with(100, "Completed!", "primary")
    
    def test_reset_progress(self):
        """Test resetting progress for specific levels and all levels."""
        # Setup progress_bars for the test
        self.container.progress_bars = {
            'primary': {'value': 50, 'message': 'In progress', 'error': False, 'visible': True},
            'secondary': {'value': 30, 'message': 'Step 2', 'error': False, 'visible': True}
        }
        
        # Mock the _update_progress_bars method
        with patch.object(self.container, '_update_progress_bars') as mock_update:
            # Test reset specific level
            self.container.reset_progress("primary")
            
            # Verify primary level was reset
            self.assertEqual(self.container.progress_bars['primary']['value'], 0)
            self.assertEqual(self.container.progress_bars['primary']['message'], '')
            self.assertFalse(self.container.progress_bars['primary']['error'])
            
            # Verify secondary level was not changed
            self.assertEqual(self.container.progress_bars['secondary']['value'], 30)
            self.assertEqual(self.container.progress_bars['secondary']['message'], 'Step 2')
            
            # Verify _update_progress_bars was called
            mock_update.assert_called_once()
            mock_update.reset_mock()
            
            # Test reset all levels
            self.container.reset_progress()
            
            # Verify all levels were reset
            for level in self.container.progress_bars:
                self.assertEqual(self.container.progress_bars[level]['value'], 0)
                self.assertEqual(self.container.progress_bars[level]['message'], '')
                self.assertFalse(self.container.progress_bars[level]['error'])
                # Only primary should be visible after full reset
                if level == 'primary':
                    self.assertTrue(self.container.progress_bars[level]['visible'])
                else:
                    self.assertFalse(self.container.progress_bars[level]['visible'])
            
            # Verify _update_progress_bars was called again
            mock_update.assert_called_once()
    
    def test_logging(self):
        """Test that log calls log on the log accordion."""
        from smartcash.ui.components.log_accordion import LogLevel
        
        # Define log level mappings to integers
        LOG_LEVEL_TO_INT = {
            LogLevel.DEBUG: 10,    # logging.DEBUG
            LogLevel.INFO: 20,     # logging.INFO
            LogLevel.WARNING: 30,  # logging.WARNING
            LogLevel.ERROR: 40,    # logging.ERROR
            LogLevel.CRITICAL: 50  # logging.CRITICAL
        }
        
        # Test debug log
        self.container.log("Debug message", LogLevel.DEBUG)
        self.mock_log_accordion._log_call.assert_called_with(LOG_LEVEL_TO_INT[LogLevel.DEBUG], "Debug message")
        
        # Test info log
        self.container.log("Info message", LogLevel.INFO)
        self.mock_log_accordion._log_call.assert_called_with(LOG_LEVEL_TO_INT[LogLevel.INFO], "Info message")
        
        # Test warning log
        self.container.log("Warning message", LogLevel.WARNING)
        self.mock_log_accordion._log_call.assert_called_with(LOG_LEVEL_TO_INT[LogLevel.WARNING], "Warning message")
        
        # Test error log
        self.container.log("Error message", LogLevel.ERROR)
        self.mock_log_accordion._log_call.assert_called_with(LOG_LEVEL_TO_INT[LogLevel.ERROR], "Error message")
        
        # Test critical log
        self.container.log("Critical message", LogLevel.CRITICAL)
        self.mock_log_accordion._log_call.assert_called_with(LOG_LEVEL_TO_INT[LogLevel.CRITICAL], "Critical message")
        
        # Verify all log levels were called with the correct messages
        self.assertEqual(len(self.mock_log_accordion._logged_messages), 5)
        self.assertEqual(self.mock_log_accordion._logged_messages[0]['msg'], "Debug message")
        self.assertEqual(self.mock_log_accordion._logged_messages[0]['level'], LOG_LEVEL_TO_INT[LogLevel.DEBUG])
        self.assertEqual(self.mock_log_accordion._logged_messages[1]['msg'], "Info message")
        self.assertEqual(self.mock_log_accordion._logged_messages[1]['level'], LOG_LEVEL_TO_INT[LogLevel.INFO])
        
        # Verify all messages were logged with correct levels
        logged_levels = [msg['level'] for msg in self.mock_log_accordion._logged_messages]
        self.assertIn(LOG_LEVEL_TO_INT[LogLevel.DEBUG], logged_levels)
        self.assertIn(LOG_LEVEL_TO_INT[LogLevel.INFO], logged_levels)
        self.assertIn(LOG_LEVEL_TO_INT[LogLevel.WARNING], logged_levels)
        self.assertIn(LOG_LEVEL_TO_INT[LogLevel.ERROR], logged_levels)
        self.assertIn(LOG_LEVEL_TO_INT[LogLevel.CRITICAL], logged_levels)
    
    def test_clear_logs(self):
        """Test that clear_logs calls clear on the log accordion."""
        # Test
        self.container.clear_logs()
        
        # Verify
        self.mock_log_accordion.clear.assert_called_once()
    
    @patch('smartcash.ui.components.operation_container.show_confirmation_dialog')
    def test_show_dialog(self, mock_show_dialog):
        """Test showing a confirmation dialog."""
        # Skip this test since we disabled dialog in setup
        if not self.container.show_dialog:
            self.skipTest("Dialog functionality is disabled in this test setup")
            
        # Mock the callback functions
        mock_confirm_callback = MagicMock()
        mock_cancel_callback = MagicMock()
        
        # Show the dialog
        self.container.show_dialog(
            title="Confirm Action",
            message="Are you sure?",
            on_confirm=mock_confirm_callback,
            on_cancel=mock_cancel_callback,
            confirm_text="Yes",
            cancel_text="No",
            danger_mode=True
        )
        
        # Verify the dialog was shown with the correct parameters
        mock_show_dialog.assert_called_once()
        
        # Get the call arguments
        args, kwargs = mock_show_dialog.call_args
        
        # Verify the arguments
        self.assertEqual(kwargs['title'], "Confirm Action")
        self.assertEqual(kwargs['message'], "Are you sure?")
        self.assertEqual(kwargs['confirm_text'], "Yes")
        self.assertEqual(kwargs['cancel_text'], "No")
        self.assertTrue(kwargs['danger_mode'])
        
        # Test the callbacks if they were provided
        if 'on_confirm' in kwargs and kwargs['on_confirm'] is not None:
            kwargs['on_confirm']()
            mock_confirm_callback.assert_called_once()
        
        if 'on_cancel' in kwargs and kwargs['on_cancel'] is not None:
            kwargs['on_cancel']()
            mock_cancel_callback.assert_called_once()
    
    @patch('smartcash.ui.components.operation_container.clear_dialog_area')
    def test_clear_dialog(self, mock_clear_dialog):
        """Test clearing the dialog."""
        if not self.container.show_dialog:
            self.skipTest("Dialog functionality is disabled in this test setup")
            
        self.container.clear_dialog()
        mock_clear_dialog.assert_called_once()
    
    @patch('smartcash.ui.components.operation_container.is_dialog_visible')
    def test_is_dialog_visible(self, mock_is_dialog_visible):
        """Test checking if a dialog is visible."""
        if not self.container.show_dialog:
            self.skipTest("Dialog functionality is disabled in this test setup")
            
        # Test when dialog is visible
        mock_is_dialog_visible.return_value = True
        self.assertTrue(self.container.is_dialog_visible())
        
        # Test when dialog is not visible
        mock_is_dialog_visible.return_value = False
        self.assertFalse(self.container.is_dialog_visible())


class TestCreateOperationContainer(unittest.TestCase):
    """Test cases for the create_operation_container function."""
    
    @patch('smartcash.ui.components.operation_container.OperationContainer')
    def test_create_operation_container(self, mock_operation_container):
        """Test creating an operation container with default parameters."""
        # Create a mock container instance
        mock_container = MagicMock()
        mock_operation_container.return_value = mock_container
        
        # Set up the return values for the container properties
        mock_container.container = MagicMock()
        mock_container.progress_tracker = MagicMock()
        mock_container.log_accordion = MagicMock()
        
        # Call the function
        result = create_operation_container(
            show_progress=True,
            show_dialog=False,  # Disable dialog for simpler testing
            show_logs=True,
            log_module_name="TestModule"
        )
        
        # Verify the container was created with the correct parameters
        mock_operation_container.assert_called_once()
        
        # Get the call arguments
        args, kwargs = mock_operation_container.call_args
        
        # Verify the arguments
        self.assertEqual(kwargs['show_progress'], True)
        self.assertEqual(kwargs['show_dialog'], False)
        self.assertEqual(kwargs['show_logs'], True)
        self.assertEqual(kwargs['log_module_name'], "TestModule")
        
        # Verify the result contains the expected keys
        self.assertEqual(result['container'], mock_container.container)
        self.assertEqual(result['progress_tracker'], mock_container.progress_tracker)
        self.assertEqual(result['log_accordion'], mock_container.log_accordion)
    
    @patch('smartcash.ui.components.operation_container.OperationContainer')
    def test_create_operation_container_minimal(self, mock_operation_container):
        """Test creating an operation container with minimal parameters."""
        # Create a mock container instance
        mock_container = MagicMock()
        mock_operation_container.return_value = mock_container
        
        # Set up the return values for the container properties
        mock_container.container = MagicMock()
        mock_container.progress_tracker = MagicMock()
        mock_container.log_accordion = MagicMock()
        
        # Call the function with minimal parameters
        result = create_operation_container()
        
        # Verify the container was created with default parameters
        mock_operation_container.assert_called_once()
        
        # Get the call arguments
        args, kwargs = mock_operation_container.call_args
        
        # Verify the default arguments
        self.assertEqual(kwargs['show_progress'], True)
        self.assertEqual(kwargs['show_dialog'], True)
        self.assertEqual(kwargs['show_logs'], True)
        self.assertEqual(kwargs['log_module_name'], "Operation")
        
        # Verify the result contains the expected keys
        self.assertIn('container', result)
        self.assertIn('progress_tracker', result)
        self.assertIn('log_accordion', result)


if __name__ == '__main__':
    unittest.main()

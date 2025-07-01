"""
Tests for the BaseHandler class in smartcash/ui/handlers/base_handler.py

This test suite validates all shared handler functionalities including:
- Logging
- Error handling
- Confirmation dialogs
- Button state management
- Status panel updates
- Progress tracking
- UI output clearing
"""

import unittest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from datetime import datetime, timedelta
import time
import logging
import sys
import io
from typing import Dict, Any

# Import the BaseHandler class
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.handlers.error_handler import ErrorContext
from smartcash.ui.utils.fallback_utils import safe_execute


class TestBaseHandler(unittest.TestCase):
    """Test suite for BaseHandler class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a concrete implementation of the abstract BaseHandler
        class ConcreteHandler(BaseHandler):
            def __init__(self):
                super().__init__("test_handler")
        
        self.handler = ConcreteHandler()
        
        # Mock UI components
        self.ui_components = {
            'status_panel': MagicMock(),
            'progress_bar': MagicMock(),
            'progress_message': MagicMock(),
            'progress_tracker': MagicMock(),
            'log_output': MagicMock(),
            'log_accordion': MagicMock(),
            'confirmation_area': MagicMock(),
            'error_area': MagicMock(),
            'output': MagicMock(),
            'button': MagicMock(),
            'button_group': [MagicMock(), MagicMock()],
        }
        
        # Set up widget properties
        for key, widget in self.ui_components.items():
            if key != 'button_group':
                widget.layout = MagicMock()
                widget.value = ""
                widget.clear_output = MagicMock()
                
        # Mock the logger to avoid addHandler issues
        self.handler.logger = MagicMock()
        
    def test_initialization(self):
        """Test initialization of BaseHandler."""
        self.assertEqual(self.handler.module_name, "test_handler")
        self.assertIsNotNone(self.handler.logger)
    
    def test_update_status_panel(self):
        """Test status panel update functionality."""
        # Test successful update
        with patch('smartcash.ui.components.status_panel.update_status_panel') as mock_update:
            # Mock status panel component
            self.ui_components['status_panel'] = MagicMock()
            
            # Call the update_status_panel method
            self.handler.update_status_panel(self.ui_components, "Test message", "info")
            
            # Verify update_status_panel was called with correct parameters
            mock_update.assert_called_once_with(self.ui_components['status_panel'], "Test message", "info")
        
        # Test graceful handling when status_panel component is missing
        del self.ui_components['status_panel']
        
        # This should not raise an exception
        self.handler.update_status_panel(self.ui_components, "Another message", "warning")
    
    def test_show_confirmation_dialog(self):
        """Test confirmation dialog functionality."""
        # Mock confirmation dialog module
        with patch('smartcash.ui.components.dialog.confirmation_dialog.show_confirmation_dialog') as mock_show:
            # Set up a simple callback function
            callback_fn = lambda: None
            
            # Test show confirmation dialog
            self.handler.show_confirmation_dialog(
                self.ui_components,
                "Are you sure?",
                callback=callback_fn,
                timeout_seconds=30
            )
            
            # Verify dialog was shown
            mock_show.assert_called_once()
            
            # Set up confirmation state manually for testing
            self.handler._confirmation_state = {
                'pending': True,
                'message': "Are you sure?",
                'timestamp': datetime.now(),
                'timeout_seconds': 30,  
                'callback': callback_fn
            }
            
            # Verify confirmation is pending
            self.assertTrue(self.handler.is_confirmation_pending(self.ui_components))
            
            # Test confirmation timeout
            self.handler._confirmation_state['timestamp'] = datetime.now() - timedelta(seconds=60)
            self.assertFalse(self.handler.is_confirmation_pending(self.ui_components))
    
    def test_show_info_dialog(self):
        """Test info dialog functionality."""
        with patch('smartcash.ui.components.dialog.confirmation_dialog.show_info_dialog') as mock_dialog:
            self.handler.show_info_dialog(
                self.ui_components,
                "Information message",
                "Info Title"
            )
            mock_dialog.assert_called_once()
    
    def test_button_state_management(self):
        """Test button state management."""
        # Create proper button mocks that work with the BaseHandler implementation
        button_mock = MagicMock()
        button_mock.disabled = False
        button_mock.description = "Test Button"  # Add description to match button detection
        
        # Add button to UI components with a name that will be detected
        self.ui_components['test_button'] = button_mock
        
        # Create individual buttons with proper naming convention to be detected
        btn1 = MagicMock()
        btn1.disabled = False
        btn1.description = "Button 1"
        self.ui_components['btn1'] = btn1
        
        btn2 = MagicMock()
        btn2.disabled = False
        btn2.description = "Button 2"
        self.ui_components['btn2'] = btn2
        
        # Test disable buttons with explicit button keys
        button_keys = ['test_button', 'btn1', 'btn2']
        self.handler.disable_all_buttons(self.ui_components, button_keys)
        
        # Verify buttons are disabled
        self.assertEqual(self.ui_components['test_button'].disabled, True)
        self.assertEqual(self.ui_components['btn1'].disabled, True)
        self.assertEqual(self.ui_components['btn2'].disabled, True)
        
        # Reset button states for enable test
        self.ui_components['test_button'].disabled = True
        self.ui_components['btn1'].disabled = True
        self.ui_components['btn2'].disabled = True
        
        # Test enable buttons with explicit button keys
        self.handler.enable_all_buttons(self.ui_components, button_keys)
        
        # Verify buttons are enabled
        self.assertEqual(self.ui_components['test_button'].disabled, False)
        self.assertEqual(self.ui_components['btn1'].disabled, False)
        self.assertEqual(self.ui_components['btn2'].disabled, False)
    
    def test_clear_ui_outputs(self):
        """Test UI output clearing."""
        # Mock log_accordion with clear_logs method
        self.ui_components['log_accordion'].clear_logs = MagicMock()
        
        # Test clearing
        self.handler.clear_ui_outputs(self.ui_components)
        
        # Verify clear_logs was called
        self.ui_components['log_accordion'].clear_logs.assert_called_once()
        
        # Verify clear_output was called on standard widgets
        self.ui_components['output'].clear_output.assert_called_once()
    
    def test_reset_progress_bars(self):
        """Test progress bar reset."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test reset
        self.handler.reset_progress_bars(self.ui_components)
        
        # Verify reset was called
        tracker_mock.reset.assert_called_once()
        
        # Test standard progress bar
        self.ui_components['progress_bar'].layout.visibility = 'visible'
        self.handler.reset_progress_bars(self.ui_components)
        self.assertEqual(self.ui_components['progress_bar'].layout.visibility, 'hidden')
    
    def test_update_progress(self):
        """Test progress update."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        # Create a ui_manager mock with is_visible property
        ui_manager_mock = MagicMock()
        ui_manager_mock.is_visible = False
        tracker_mock.ui_manager = ui_manager_mock
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test update
        self.handler.update_progress(
            self.ui_components, 
            50, 
            100, 
            "Processing...", 
            "primary"
        )
        
        # Verify tracker methods were called
        tracker_mock.show.assert_called_once()
        tracker_mock.update_primary.assert_called_once_with(50, "Processing...")
        
        # Test standard progress bar
        del self.ui_components['progress_tracker']
        self.handler.update_progress(self.ui_components, 75, 100, "Almost done")
        self.assertEqual(self.ui_components['progress_bar'].value, 75)
    
    def test_complete_progress(self):
        """Test progress completion."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test complete
        self.handler.complete_progress(self.ui_components, "Operation successful!")
        
        # Verify tracker methods were called
        tracker_mock.complete.assert_called_once_with("Operation successful!")
        
        # Test standard progress bar
        del self.ui_components['progress_tracker']
        self.handler.complete_progress(self.ui_components, "Done!")
        self.assertEqual(self.ui_components['progress_bar'].value, 100)
    
    def test_error_progress(self):
        """Test progress error."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test error
        self.handler.error_progress(self.ui_components, "Operation failed!")
        
        # Verify tracker methods were called
        tracker_mock.error.assert_called_once_with("Operation failed!")
    
    def test_log_redirection(self):
        """Test log redirection to log_accordion."""
        # Create a custom _update_log_accordion method for LogAccordionHandler
        def custom_update(ui_components, message, level):
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
                ui_components['log_output'].append_log(message=message, level=level)
            elif 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'append_log'):
                ui_components['log_accordion'].append_log(message=message, level=level)
        
        # Mock log_output with append_log method
        self.ui_components['log_output'] = MagicMock()
        self.ui_components['log_output'].append_log = MagicMock()
        
        # Set up log redirection
        self.handler.setup_log_redirection(self.ui_components)
        
        # Capture the log handler and add our custom update method
        log_handler = self.handler._log_handler
        log_handler._update_log_accordion = custom_update
        self.assertIsNotNone(log_handler)
        
        # Create a test log record
        record = logging.LogRecord(
            name=self.handler.logger.name,
            level=logging.INFO,
            pathname="test_file.py",
            lineno=1,
            msg="Test log message",
            args=(),
            exc_info=None
        )
        
        # Process the record through our handler
        log_handler.emit(record)
        
        # Verify log_output.append_log was called
        self.ui_components['log_output'].append_log.assert_called_once()
        
        # Test direct update to log_accordion
        self.ui_components['log_output'].append_log.reset_mock()
        self.handler._update_log_accordion(self.ui_components, "Direct message", "info")
        
        # Verify log_output.append_log was called again
        self.ui_components['log_output'].append_log.assert_called_once_with(
            message="Direct message", level="info")
        
        # Test fallback to log_accordion when log_output is not available
        del self.ui_components['log_output']
        self.ui_components['log_accordion'] = MagicMock()
        self.ui_components['log_accordion'].append_log = MagicMock()
        
        # Test direct update to log_accordion
        self.handler._update_log_accordion(self.ui_components, "Fallback message", "warning")
        
        # Verify log_accordion.append_log was called
        self.ui_components['log_accordion'].append_log.assert_called_once_with(
            message="Fallback message", level="warning")
    
    def test_single_progress_wrapper(self):
        """Test single progress tracker wrapper."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        # Create a ui_manager mock with is_visible property
        ui_manager_mock = MagicMock()
        ui_manager_mock.is_visible = False
        tracker_mock.ui_manager = ui_manager_mock
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test update
        self.handler.update_single_progress(
            self.ui_components, 
            50, 
            100, 
            "Processing..."
        )
        
        # Verify tracker methods were called
        tracker_mock.show.assert_called_once()
        tracker_mock.update_primary.assert_called_once_with(50, "Processing...")
        
        # Test fallback to standard progress update when tracker is not available
        del self.ui_components['progress_tracker']
        with patch.object(self.handler, 'update_progress') as mock_update:
            self.handler.update_single_progress(self.ui_components, 75, 100, "Almost done")
            mock_update.assert_called_once_with(self.ui_components, 75, 100, "Almost done")
    
    def test_dual_progress_wrapper(self):
        """Test dual progress tracker wrapper."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        # Create a ui_manager mock with is_visible property
        ui_manager_mock = MagicMock()
        ui_manager_mock.is_visible = False
        tracker_mock.ui_manager = ui_manager_mock
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test update
        self.handler.update_dual_progress(
            self.ui_components, 
            50, 100,  # overall value, max
            30, 100,  # current value, max
            "Processing...",  # overall message
            "Current task"  # current message
        )
        
        # Verify tracker methods were called
        tracker_mock.show.assert_called_once()
        # Check that update methods were called with the correct parameters
        tracker_mock.update_primary.assert_called_once_with(50, "Processing...")
        tracker_mock.update_current.assert_called_once_with(30, "Current task")
        
        # Test fallback to standard progress update when tracker is not available
        del self.ui_components['progress_tracker']
        with patch.object(self.handler, 'update_progress') as mock_update:
            self.handler.update_dual_progress(
                self.ui_components, 
                75, 100,  # overall value, max
                60, 100,  # current value, max
                "Almost done",  # overall message
                "Secondary"  # current message
            )
            mock_update.assert_called_once_with(self.ui_components, 75, 100, "Almost done")

    def test_triple_progress_wrapper(self):
        """Test triple progress tracker wrapper."""
        # Mock progress_tracker
        from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
        tracker_mock = MagicMock(spec=ProgressTracker)
        # Create a ui_manager mock with is_visible property
        ui_manager_mock = MagicMock()
        ui_manager_mock.is_visible = False
        tracker_mock.ui_manager = ui_manager_mock
        self.ui_components['progress_tracker'] = tracker_mock
        
        # Test update
        self.handler.update_triple_progress(
            self.ui_components, 
            30, 100,  # overall value, max
            5, 10,    # current value, max
            2, 5,     # step value, max
            "Overall progress",  # overall message
            "Current step",     # current message
            "Step detail"       # step message
        )
        
        # Verify tracker methods were called
        tracker_mock.show.assert_called_once()
        # Check that update methods were called with the correct parameters
        tracker_mock.update_primary.assert_called_once_with(30, "Overall progress")
        tracker_mock.update_current.assert_called_once_with(5, "Current step")
        tracker_mock.update_step.assert_called_once_with(2, "Step detail")
        
        # Test fallback to standard progress update when tracker is not available
        del self.ui_components['progress_tracker']
        with patch.object(self.handler, 'update_progress') as mock_update:
            self.handler.update_triple_progress(
                self.ui_components, 
                40, 100,  # overall value, max
                6, 10,    # current value, max
                3, 5,     # step value, max
                "Overall progress",  # overall message
                "Current step",     # current message
                "Step detail"       # step message
            )
            mock_update.assert_called_once_with(
                self.ui_components, 40, 100, "Overall progress"
            )
        
        # No error response test here - it belongs in a separate test method
    
    def test_busy_cursor_context_manager(self):
        """Test busy cursor context manager."""
        # Test context manager
        with self.handler.with_busy_cursor(self.ui_components) as is_busy:
            self.assertTrue(is_busy)
            self.assertTrue(self.ui_components['button'].disabled)
        
        # After context, buttons should be enabled
        self.assertFalse(self.ui_components['button'].disabled)
    
    def test_safe_execute(self):
        """Test safe execution of functions."""
        # Test successful execution using imported safe_execute
        result = safe_execute(lambda x, y: x + y, x=1, y=2)
        self.assertEqual(result, 3)
        
        # Test error handling
        result = safe_execute(lambda: 1/0)
        self.assertIsNone(result)
    
    def test_api_response_checking(self):
        """Test API response checking."""
        # Test successful response with 'status' key
        response = {"status": "success", "data": "test"}
        self.assertTrue(self.handler.is_success_response(response))
        
        # Test error response with 'status' key
        response = {"status": "error", "message": "Failed"}
        self.assertFalse(self.handler.is_success_response(response))
        
        # Test missing status key
        response = {"data": "test"}
        self.assertFalse(self.handler.is_success_response(response))
        
        # Test with old 'success' key (should not work)
        response = {"success": True, "data": "test"}
        self.assertFalse(self.handler.is_success_response(response))
        
        # Test with both keys (should use 'status')
        response = {"status": "success", "success": False, "data": "test"}
        self.assertTrue(self.handler.is_success_response(response))
        
        # Test non-dict response
        response = "Not a dict"
        self.assertFalse(self.handler.is_success_response(response))


if __name__ == '__main__':
    unittest.main()

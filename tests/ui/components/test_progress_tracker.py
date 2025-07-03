"""Tests for the ProgressTracker component."""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from datetime import datetime

from smartcash.ui.components.progress_tracker.progress_tracker import (
    ProgressTracker,
    create_progress_tracker,
    update_progress,
    complete_progress,
    error_progress
)
from smartcash.ui.components.progress_tracker.progress_config import (
    ProgressLevel,
    ProgressBarConfig,
    ProgressConfig
)


class TestProgressTracker(unittest.TestCase):
    """Test cases for the ProgressTracker component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock ProgressConfig
        self.mock_config = MagicMock(spec=ProgressConfig)
        self.mock_config.get_level_configs.return_value = [
            ProgressBarConfig("overall", "Overall Progress", "", "#28a745", 0, True),
            ProgressBarConfig("step", "Step Progress", "", "#28a745", 1, True),
            ProgressBarConfig("current", "Current Operation", "", "#28a745", 2, True)
        ]
        self.mock_config.get_container_height.return_value = "200px"
        self.mock_config.level = ProgressLevel.TRIPLE
        
        # Create a ProgressTracker instance for testing
        self.tracker = ProgressTracker(config=self.mock_config)
        
        # Mock the TqdmManager and CallbackManager
        self.mock_tqdm_manager = MagicMock()
        self.mock_callback_manager = MagicMock()
        
        # Set up the mock callback manager to return a mock when register is called
        self.mock_callback_manager.register.side_effect = lambda event, cb: f"{event}_id"
        
        # Patch the TqdmManager and CallbackManager in the tracker
        self.tracker.tqdm_manager = self.mock_tqdm_manager
        self.tracker.callback_manager = self.mock_callback_manager
        
        # Mock the widgets and their layouts
        self.mock_container = MagicMock(spec=widgets.VBox)
        self.mock_container.layout = MagicMock()
        self.mock_header = MagicMock(spec=widgets.HTML)
        self.mock_status = MagicMock(spec=widgets.HTML)
        
        # Set up the layout update mock to store calls for assertions
        self.layout_updates = []
        def record_update(**kwargs):
            self.layout_updates.append(kwargs)
        self.mock_container.layout.update.side_effect = record_update
        
        # Set up the _ui_components dictionary
        self.tracker._ui_components = {
            'container': self.mock_container,
            'header': self.mock_header,
            'status': self.mock_status,
            'overall_output': MagicMock(spec=widgets.Output),
            'step_output': MagicMock(spec=widgets.Output),
            'current_output': MagicMock(spec=widgets.Output)
        }
        
        # Mark as initialized
        self.tracker._initialized = True
    
    def test_initialization(self):
        """Test that the ProgressTracker initializes correctly."""
        self.assertEqual(self.tracker._config, self.mock_config)
        self.assertIsNotNone(self.tracker.callback_manager)
        self.assertEqual(self.tracker._active_levels, ['overall', 'step', 'current'])
        self.assertEqual(self.tracker._current_step_index, 0)
        self.assertFalse(self.tracker._is_complete)
        self.assertFalse(self.tracker._is_error)
    
    def test_show(self):
        """Test that the show method updates the container display."""
        # Reset the mock layout
        self.mock_container.layout = MagicMock()
        
        # Call the method being tested
        self.tracker.show()
        
        # Check that the layout properties were set directly
        self.assertEqual(self.mock_container.layout.display, 'flex')
        self.assertEqual(self.mock_container.layout.visibility, 'visible')
    
    def test_set_progress(self):
        """Test setting progress for a specific level."""
        # Reset the mock to clear any previous calls
        self.mock_tqdm_manager.update_bar.reset_mock()
        
        # Test with valid level
        self.tracker.set_progress(50, "overall", "Halfway there!")
        self.mock_tqdm_manager.update_bar.assert_called_once_with("overall", 50, "Halfway there!")
        
        # Reset the mock for the next test
        self.mock_tqdm_manager.update_bar.reset_mock()
        
        # Test with invalid level - should still call update_bar but with the invalid level
        self.tracker.set_progress(50, "invalid", "Test")
        self.mock_tqdm_manager.update_bar.assert_called_once_with("invalid", 50, "Test")
    
    def test_complete(self):
        """Test completing the progress."""
        self.tracker.complete("All done!")
        self.assertTrue(self.tracker._is_complete)
        self.mock_tqdm_manager.set_all_complete.assert_called_once_with("All done!", self.mock_config.get_level_configs())
        self.mock_callback_manager.trigger.assert_called_once_with('complete')
    
    def test_error(self):
        """Test error state for progress."""
        self.tracker.error("Something went wrong!")
        self.assertTrue(self.tracker._is_error)
        self.mock_tqdm_manager.set_all_error.assert_called_once_with("Something went wrong!")
        self.mock_callback_manager.trigger.assert_called_once_with('error', "Something went wrong!")
    
    def test_reset(self):
        """Test resetting the progress."""
        # Set some state
        self.tracker._current_step_index = 5
        self.tracker._is_complete = True
        self.tracker._is_error = True
        
        # Reset the mocks
        self.mock_tqdm_manager.initialize_bars.reset_mock()
        self.mock_callback_manager.trigger.reset_mock()
        
        # Reset
        self.tracker.reset()
        
        # Check state
        self.assertEqual(self.tracker._current_step_index, 0)
        self.assertFalse(self.tracker._is_complete)
        self.assertFalse(self.tracker._is_error)
        self.mock_tqdm_manager.initialize_bars.assert_called_once_with(self.mock_config.get_level_configs())
        self.mock_callback_manager.trigger.assert_called_once_with('reset')
    
    def test_update_status(self):
        """Test updating the status message."""
        self.tracker.update_status("Test message", "status")
        self.assertEqual(self.mock_status.value, "Test message")
    
    def test_hide(self):
        """Test hiding the progress tracker."""
        # Reset the mock layout
        self.mock_container.layout = MagicMock()
        
        # Call the method being tested
        self.tracker.hide()
        
        # Check that the layout properties were set directly
        self.assertEqual(self.mock_container.layout.display, 'none')
        self.assertEqual(self.mock_container.layout.visibility, 'hidden')
    
    def test_callback_registration(self):
        """Test registering and removing callbacks."""
        # Reset the callback manager mock
        self.mock_callback_manager.register.reset_mock()
        self.mock_callback_manager.unregister.reset_mock()
        
        # Test registering callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        # Register callbacks
        id1 = self.tracker.on_progress_update(callback1)
        id2 = self.tracker.on_complete(callback2)
        
        # Verify callbacks were registered
        self.mock_callback_manager.register.assert_any_call('progress_update', callback1)
        self.mock_callback_manager.register.assert_any_call('complete', callback2)
        
        # Test removing callback
        self.tracker.remove_callback(id1)
        self.mock_callback_manager.unregister.assert_called_once_with(id1)


class TestLegacyFunctions(unittest.TestCase):
    """Test cases for legacy functions."""
    
    @patch('smartcash.ui.components.progress_tracker.progress_tracker.ProgressTracker')
    def test_create_progress_tracker(self, mock_tracker_class):
        """Test the create_progress_tracker function."""
        mock_config = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        
        # Test with config
        result = create_progress_tracker(mock_config)
        
        # Check that ProgressTracker was instantiated with the correct arguments
        # The config should be passed as a positional argument, not a keyword argument
        args, _ = mock_tracker_class.call_args
        self.assertEqual(len(args), 2)  # component_name and config
        self.assertEqual(args[1], mock_config)  # config is the second positional argument
        self.assertEqual(result, mock_tracker)
    
    def test_update_progress(self):
        """Test the update_progress function."""
        mock_tracker = MagicMock()
        update_progress(mock_tracker, 50, "Halfway!", "overall")
        mock_tracker.set_progress.assert_called_once_with(50, "overall", "Halfway!")
    
    def test_complete_progress(self):
        """Test the complete_progress function."""
        mock_tracker = MagicMock()
        complete_progress(mock_tracker, "Done!")
        mock_tracker.complete.assert_called_once_with("Done!")
    
    def test_error_progress(self):
        """Test the error_progress function."""
        mock_tracker = MagicMock()
        error_progress(mock_tracker, "Error!")
        mock_tracker.error.assert_called_once_with("Error!")


if __name__ == "__main__":
    unittest.main()

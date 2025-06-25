"""
Unit tests for DualProgressTracker class
"""
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from ipywidgets import VBox, FloatProgress, HTML

from smartcash.ui.setup.env_config.utils.dual_progress_tracker import (
    DualProgressTracker, SetupStage
)

class TestDualProgressTracker(unittest.TestCase):
    """Test cases for DualProgressTracker class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_logger = MagicMock()
        self.components = {}
        self.tracker = DualProgressTracker(ui_components=self.components, logger=self.mock_logger)

    def test_initialization(self):
        """Test that the tracker initializes correctly"""
        self.assertIsInstance(self.tracker.container, VBox)
        self.assertEqual(len(self.tracker.container.children), 2)
        self.assertIsInstance(self.tracker.container.children[0], FloatProgress)
        self.assertIsInstance(self.tracker.container.children[1], HTML)
        self.assertEqual(self.tracker.overall_progress, 0)
        self.assertEqual(self.tracker.stage_progress, 0)
        self.assertIsNone(self.tracker.current_stage)
        self.assertEqual(self.components['progress_container'], self.tracker.container)
        self.assertEqual(self.components['progress_tracker'], self.tracker)

    def test_update_stage(self):
        """Test updating the current stage"""
        self.tracker.update_stage(SetupStage.FOLDER_SETUP, "Testing folder setup")
        self.assertEqual(self.tracker.current_stage, SetupStage.FOLDER_SETUP)
        # Check that the message is in the status text (without checking exact formatting)
        self.assertIn("Testing folder setup", self.tracker.status_text.value)
        self.assertEqual(self.tracker.stage_progress, 0)

    def test_update_progress(self):
        """Test updating progress within a stage"""
        self.tracker.update_stage(SetupStage.CONFIG_SYNC)
        self.tracker.update_progress(50, "Halfway there")
        self.assertEqual(self.tracker.stage_progress, 50)
        self.assertEqual(self.tracker.progress_bar.value, 50)
        self.assertIn("Halfway there", self.tracker.status_text.value)

    def test_update_within_stage(self):
        """Test updating progress using item counts"""
        self.tracker.update_stage(SetupStage.ENV_SETUP)
        self.tracker.update_within_stage(3, 10, "Processing items")
        self.assertEqual(self.tracker.stage_progress, 30)
        self.assertIn("Processing items", self.tracker.status_text.value)
        
        # Test edge cases
        self.tracker.update_within_stage(0, 0, "Should not update")
        self.assertEqual(self.tracker.stage_progress, 30)  # Should remain unchanged
        
        self.tracker.update_within_stage(1, 1, "Single item")
        self.assertEqual(self.tracker.stage_progress, 100)

    def test_complete_stage(self):
        """Test completing the current stage"""
        self.tracker.update_stage(SetupStage.DRIVE_MOUNT)
        self.tracker.complete_stage("Mount complete")
        self.assertEqual(self.tracker.stage_progress, 100)
        self.assertEqual(self.tracker.progress_bar.value, 100)
        # Check that the message is in the status text (without checking exact prefix)
        self.assertIn("Mount complete", self.tracker.status_text.value)

    def test_complete(self):
        """Test completing the entire setup"""
        self.tracker.complete("All done!")
        self.assertEqual(self.tracker.overall_progress, 100)
        self.assertEqual(self.tracker.stage_progress, 100)
        self.assertEqual(self.tracker.progress_bar.value, 100)
        # Check that the message is in the status text (without checking exact prefix)
        self.assertIn("All done!", self.tracker.status_text.value)

    def test_error(self):
        """Test error reporting"""
        self.tracker.error("Something went wrong")
        self.assertIn("❌ Error: Something went wrong", self.tracker.status_text.value)
        self.mock_logger.error.assert_called_once_with("Something went wrong")

    def test_add_callback(self):
        """Test callback functionality"""
        callback = MagicMock()
        self.tracker.add_callback(callback)
        self.tracker.update_stage(SetupStage.SYMLINK_SETUP)
        callback.assert_called_once()
        
        # Test with non-callable
        self.tracker.add_callback("not a function")  # Should not raise
        self.assertEqual(len(self.tracker.callbacks), 1)  # Still only the first callback

    def test_progress_calculation(self):
        """Test overall progress calculation"""
        # Test with no stage set
        self.tracker._update_overall_progress()
        self.assertEqual(self.tracker.overall_progress, 0)
        
        # Test with first stage
        self.tracker.update_stage(SetupStage.DRIVE_MOUNT)
        self.tracker.update_progress(50)
        stage_count = len(SetupStage)
        expected_progress = (0 + (50 / 100) * (100 / stage_count))
        self.assertAlmostEqual(self.tracker.overall_progress, expected_progress)
        
        # Test with last stage complete
        self.tracker.update_stage(SetupStage.COMPLETE)
        self.tracker.update_progress(100)
        self.assertEqual(self.tracker.overall_progress, 100)

    def test_ui_creation(self):
        """Test UI widget creation"""
        # Just verify that the UI elements are created and connected
        self.assertIsNotNone(self.tracker.container)
        self.assertIsNotNone(self.tracker.progress_bar)
        self.assertIsNotNone(self.tracker.status_text)
        self.assertEqual(len(self.tracker.container.children), 2)
        self.assertIn(self.tracker.progress_bar, self.tracker.container.children)
        self.assertIn(self.tracker.status_text, self.tracker.container.children)

    def test_progress_container_property(self):
        """Test the progress_container property"""
        container = self.tracker.progress_container
        self.assertIs(container, self.tracker.container)

    def test_ui_components_property(self):
        """Test the ui_components property"""
        components = self.tracker.ui_components
        self.assertIs(components, self.components)
        self.assertEqual(components['progress_container'], self.tracker.container)
        self.assertEqual(components['progress_tracker'], self.tracker)
        
    def test_show_hide_visibility(self):
        """Test show() and hide() methods with visibility layout"""
        # Create a real layout with visibility
        from ipywidgets import Layout
        layout = Layout(visibility='hidden')
        self.tracker.container.layout = layout
        
        # Test show()
        self.tracker.show()
        self.assertEqual(self.tracker.container.layout.visibility, 'visible')
        
        # Test hide()
        self.tracker.hide()
        self.assertEqual(self.tracker.container.layout.visibility, 'hidden')
    
    def test_show_hide_display(self):
        """Test show() and hide() methods with display layout"""
        # Skip if container doesn't support display property
        if not (hasattr(self.tracker.container, 'layout') and 
               hasattr(self.tracker.container.layout, 'display')):
            self.skipTest("Container layout doesn't support display property")
        
        # Save original display value
        original_display = getattr(self.tracker.container.layout, 'display', None)
        
        try:
            # Test show() sets display to 'flex'
            self.tracker.show()
            self.assertEqual(self.tracker.container.layout.display, 'flex')
            
            # Hide should only affect display if visibility is not supported
            self.tracker.hide()
            if hasattr(self.tracker.container.layout, 'visibility'):
                # If visibility is supported, display should remain 'flex'
                self.assertEqual(self.tracker.container.layout.display, 'flex')
            else:
                # If no visibility, display should be set to 'none'
                self.assertEqual(self.tracker.container.layout.display, 'none')
            
            # Test show() sets display back to 'flex'
            self.tracker.show()
            self.assertEqual(self.tracker.container.layout.display, 'flex')
            
        finally:
            # Restore original display value if it existed
            if original_display is not None:
                self.tracker.container.layout.display = original_display
    
    def test_show_hide_no_layout(self):
        """Test show() and hide() methods when container has no layout"""
        # These should not raise exceptions even without layout
        self.tracker.show()
        self.tracker.hide()
        
        # Test with layout but no visibility/display attributes
        from ipywidgets import Layout
        layout = Layout()
        self.tracker.container.layout = layout
        self.tracker.show()  # Should not raise
        self.tracker.hide()  # Should not raise

if __name__ == '__main__':
    unittest.main()

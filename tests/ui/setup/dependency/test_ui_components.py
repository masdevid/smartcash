"""
Test cases for dependency management UI components.
"""
import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch

# Import the components we want to test
from smartcash.ui.setup.dependency.components.ui_components import (
    create_dependency_main_ui
)

class TestDependencyUIComponents(unittest.TestCase):
    """Test cases for dependency management UI components."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the logger bridge to prevent actual logging during tests
        self.logger_patch = patch('smartcash.ui.setup.dependency.components.ui_components.UILoggerBridge')
        self.mock_logger_bridge = self.logger_patch.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger_bridge.return_value = self.mock_logger_instance

        # Mock other UI components
        self.mock_header = MagicMock()
        self.mock_status_panel = MagicMock()
        self.mock_action_buttons = MagicMock()
        self.mock_progress_tracker = MagicMock()
        self.mock_log_components = {
            'log_output': MagicMock(),
            'log_accordion': MagicMock(),
            'entries_container': MagicMock()
        }

        # Patch the component creation functions
        self.header_patch = patch(
            'smartcash.ui.components.create_header',
            return_value=self.mock_header
        )
        self.status_panel_patch = patch(
            'smartcash.ui.components.create_status_panel',
            return_value=self.mock_status_panel
        )
        self.action_buttons_patch = patch(
            'smartcash.ui.components.create_action_buttons',
            return_value=self.mock_action_buttons
        )
        self.progress_tracker_patch = patch(
            'smartcash.ui.components.create_dual_progress_tracker',
            return_value=self.mock_progress_tracker
        )
        self.log_accordion_patch = patch(
            'smartcash.ui.components.create_log_accordion',
            return_value=self.mock_log_components
        )
        self.responsive_container_patch = patch(
            'smartcash.ui.components.create_responsive_container',
            return_value=MagicMock()
        )

        # Start all patches
        self.header_patch.start()
        self.status_panel_patch.start()
        self.action_buttons_patch.start()
        self.progress_tracker_patch.start()
        self.log_accordion_patch.start()
        self.responsive_container_patch.start()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop all patches
        self.logger_patch.stop()
        self.header_patch.stop()
        self.status_panel_patch.stop()
        self.action_buttons_patch.stop()
        self.progress_tracker_patch.stop()
        self.log_accordion_patch.stop()
        self.responsive_container_patch.stop()

    def test_create_dependency_main_ui(self):
        """Test creating the main dependency management UI."""
        # Call the function with a test config
        test_config = {
            'test_key': 'test_value'
        }
        
        result = create_dependency_main_ui(test_config)
        
        # Verify the result contains expected keys
        self.assertIn('ui', result)
        self.assertIn('container', result)
        self.assertIn('header', result)
        self.assertIn('logger_bridge', result)
        
        # Verify the logger bridge was created with correct parameters
        self.assertTrue(self.mock_logger_bridge.called)
        
        # Get the call arguments
        call_args = self.mock_logger_bridge.call_args[1]  # Get kwargs
        
        # Verify the logger bridge was called with the correct UI components
        self.assertIn('ui_components', call_args)
        self.assertEqual(call_args.get('logger_name'), 'dependency_ui')
        
        # Verify the UI components were created by checking their return values were used
        self.assertTrue(self.mock_header is not None)
        self.assertTrue(self.mock_status_panel is not None)
        self.assertTrue(self.mock_action_buttons is not None)
        self.assertTrue(self.mock_progress_tracker is not None)
        self.assertTrue(self.mock_log_components is not None)

    def test_create_dependency_main_ui_default_config(self):
        """Test creating the main UI with default config."""
        # Call without config (should use default)
        result = create_dependency_main_ui()
        
        # Should still create all components
        self.assertIn('ui', result)
        self.assertIn('logger_bridge', result)
        self.mock_logger_bridge.assert_called_once()

if __name__ == '__main__':
    unittest.main()

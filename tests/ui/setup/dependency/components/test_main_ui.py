"""
Test cases for the main dependency management UI component.
"""
import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any

class TestDependencyMainUI(unittest.TestCase):
    """Test cases for the main dependency management UI."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the logger bridge
        self.logger_patch = patch('smartcash.ui.setup.dependency.components.ui_components.UILoggerBridge')
        self.mock_logger_bridge = self.logger_patch.start()
        self.mock_logger_instance = MagicMock()
        self.mock_logger_bridge.return_value = self.mock_logger_instance

        # Create proper widget mocks that implement the ipywidgets interface
        def create_widget_mock(**kwargs):
            mock = MagicMock()
            mock.layout = MagicMock()
            for k, v in kwargs.items():
                setattr(mock, k, v)
            return mock
            
        # Mock UI components with proper widget mocks
        self.mock_header = create_widget_mock()
        self.mock_status_panel = create_widget_mock()
        self.mock_action_buttons = create_widget_mock()
        self.mock_progress_tracker = create_widget_mock(container=create_widget_mock())
        
        # Mock log components
        self.mock_log_output = MagicMock(spec=widgets.Output)
        self.mock_entries_container = MagicMock(spec=widgets.VBox)
        self.mock_log_accordion = MagicMock(spec=widgets.Accordion)
        
        # Set up the children structure
        self.mock_entries_container.children = [self.mock_log_output]
        self.mock_log_accordion.children = [self.mock_entries_container]
        
        # Create the log components dictionary
        self.mock_log_components = {
            'log_output': self.mock_log_output,
            'log_accordion': self.mock_log_accordion,
            'entries_container': self.mock_entries_container
        }
        
        # Mock other UI sections
        self.mock_custom_section = create_widget_mock()
        self.mock_categories_section = create_widget_mock()
        self.mock_action_section = create_widget_mock()
        self.mock_confirmation_area = create_widget_mock()

        # Patch the component creation functions
        self.patches = {
            'header': patch('smartcash.ui.components.create_header', return_value=self.mock_header),
            'status_panel': patch('smartcash.ui.components.create_status_panel', return_value=self.mock_status_panel),
            'action_buttons': patch('smartcash.ui.components.create_action_buttons'),
            'progress_tracker': patch('smartcash.ui.components.create_dual_progress_tracker', 
                                    return_value=self.mock_progress_tracker),
            'log_accordion': patch('smartcash.ui.components.create_log_accordion', 
                                 return_value=self.mock_log_components),
            'module_log_accordion': patch('smartcash.ui.setup.dependency.components.ui_components.create_log_accordion',
                                       return_value=self.mock_log_components),
            'responsive_container': patch('smartcash.ui.components.create_responsive_container',
                                       side_effect=lambda children, **kwargs: MagicMock(children=children)),
            'text_input': patch('smartcash.ui.components.create_text_input', return_value=MagicMock()),
            'confirmation_area': patch('smartcash.ui.components.create_confirmation_area',
                                     return_value=(self.mock_confirmation_area, MagicMock())),
            'action_section': patch('smartcash.ui.components.create_action_section',
                                 return_value=self.mock_action_section),
            'checkbox': patch('ipywidgets.Checkbox', return_value=MagicMock()),
            'html': patch('ipywidgets.HTML', return_value=MagicMock()),
            'vbox': patch('ipywidgets.VBox', side_effect=lambda children=None, **kwargs: MagicMock(children=children or [])),
            'categories_section': patch('smartcash.ui.setup.dependency.components.ui_components._create_categories_section',
                                     return_value=self.mock_categories_section),
            'custom_section': patch('smartcash.ui.setup.dependency.components.ui_components._create_custom_package_section',
                                 return_value=self.mock_custom_section),
            'default_config': patch('smartcash.ui.setup.dependency.handlers.defaults.get_default_dependency_config',
                                 return_value={'categories': []})
        }
        
        # Start all patches
        for patch_obj in self.patches.values():
            patch_obj.start()
            
        # Set up action buttons
        self.setup_action_buttons()
        
    def setup_action_buttons(self):
        """Set up mock action buttons for testing."""
        self.mock_install_btn = MagicMock(description='install_btn')
        self.mock_check_updates_btn = MagicMock(description='check_updates_btn')
        self.mock_uninstall_btn = MagicMock(description='uninstall_btn')
        
        # Create a proper HBox mock for action buttons with children
        self.mock_action_buttons = MagicMock(spec=widgets.HBox)
        self.mock_action_buttons.children = [
            self.mock_install_btn,
            self.mock_check_updates_btn,
            self.mock_uninstall_btn
        ]
        
        # Configure the action buttons patch
        self.patches['action_buttons'].return_value = self.mock_action_buttons

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop all patches in reverse order
        for patch_obj in reversed(self.patches.values()):
            patch_obj.stop()
        self.logger_patch.stop()

    def test_create_dependency_main_ui(self):
        """Test creating the main dependency management UI."""
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        
        # Create a test config
        test_config = {
            'categories': [{
                'name': 'Test Category',
                'description': 'Test Description',
                'icon': '📦',
                'packages': [{
                    'name': 'test-package',
                    'description': 'Test package',
                    'key': 'test-pkg',
                    'pip_name': 'test-package',
                    'required': True,
                    'installed': False,
                    'version': '1.0.0',
                    'latest_version': '1.0.0',
                    'update_available': False,
                    'dependencies': []
                }]
            }]
        }
        
        # Call the function with test config
        result = create_dependency_main_ui(test_config)
        
        # Verify the result contains all expected keys
        expected_keys = [
            'ui', 'container', 'header', 'status_panel', 'categories_section',
            'custom_section', 'action_section', 'progress_tracker',
            'log_components', 'log_output', 'logger_bridge',
            'install_btn', 'check_updates_btn', 'uninstall_btn'
        ]
        
        # Check for required keys
        for key in expected_keys:
            self.assertIn(key, result, f"Expected key '{key}' not found in result")
        
        # Verify the UI components are correctly set
        self.assertIs(result['ui'], result['container'])
        self.assertIs(result['header'], self.mock_header)
        self.assertIs(result['status_panel'], self.mock_status_panel)
        self.assertIs(result['categories_section'], self.mock_categories_section)
        self.assertIs(result['custom_section'], self.mock_custom_section)
        self.assertIs(result['action_section'], self.mock_action_section)
        self.assertIs(result['progress_tracker'], self.mock_progress_tracker)
        self.assertIs(result['log_components'], self.mock_log_components)
        self.assertIs(result['log_output'], self.mock_log_components['log_output'])
        self.assertIs(result['logger_bridge'], self.mock_logger_instance)
        
        # Verify action buttons
        self.assertIs(result['install_btn'], self.mock_install_btn)
        self.assertIs(result['check_updates_btn'], self.mock_check_updates_btn)
        self.assertIs(result['uninstall_btn'], self.mock_uninstall_btn)

if __name__ == '__main__':
    unittest.main()

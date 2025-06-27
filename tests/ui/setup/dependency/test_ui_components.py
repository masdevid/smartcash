"""
Test cases for dependency management UI components.
"""
import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, List, Optional

# Import the components we want to test
from smartcash.ui.setup.dependency.components.ui_components import (
    create_dependency_main_ui,
    _create_categories_section,
    _create_custom_package_section,
    _create_package_checkbox,
    _extract_category_components,
    _extract_custom_components,
    update_package_status
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

        # Create proper widget mocks that implement the ipywidgets interface
        def create_widget_mock(**kwargs):
            mock = MagicMock()
            mock.layout = MagicMock()
            for k, v in kwargs.items():
                setattr(mock, k, v)
            return mock
            
        # Mock other UI components with proper widget mocks
        self.mock_header = create_widget_mock()
        self.mock_status_panel = create_widget_mock()
        self.mock_action_buttons = create_widget_mock()
        self.mock_progress_tracker = create_widget_mock(container=create_widget_mock())
        
        # Create a proper mock for the log components that will be returned by create_log_accordion
        self.mock_log_output = MagicMock(spec=widgets.Output)
        self.mock_entries_container = MagicMock(spec=widgets.VBox)
        self.mock_log_accordion = MagicMock(spec=widgets.Accordion)
        
        # Set up the children structure
        self.mock_entries_container.children = [self.mock_log_output]
        self.mock_log_accordion.children = [self.mock_entries_container]
        
        # Create the log components dictionary with proper widget mocks
        self.mock_log_components = {
            'log_output': self.mock_log_output,
            'log_accordion': self.mock_log_accordion,
            'entries_container': self.mock_entries_container
        }
        
        self.mock_custom_section = create_widget_mock()
        self.mock_categories_section = create_widget_mock()
        self.mock_action_section = create_widget_mock()
        self.mock_confirmation_area = create_widget_mock()

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
        # Patch create_log_accordion to return our mock components
        def mock_create_log_accordion(*args, **kwargs):
            return self.mock_log_components
            
        self.log_accordion_patch = patch(
            'smartcash.ui.components.create_log_accordion',
            side_effect=mock_create_log_accordion
        )
        
        # Patch the module-level import as well
        self.module_log_accordion_patch = patch(
            'smartcash.ui.setup.dependency.components.ui_components.create_log_accordion',
            side_effect=mock_create_log_accordion
        )
        self.responsive_container_patch = patch(
            'smartcash.ui.components.create_responsive_container',
            side_effect=lambda children, **kwargs: MagicMock(children=children)
        )
        self.custom_section_patch = patch(
            'smartcash.ui.components.create_text_input',
            return_value=MagicMock()
        )
        self.confirmation_area_patch = patch(
            'smartcash.ui.components.create_confirmation_area',
            return_value=(self.mock_confirmation_area, MagicMock())
        )
        self.action_section_patch = patch(
            'smartcash.ui.components.create_action_section',
            return_value=self.mock_action_section
        )
        self.checkbox_patch = patch(
            'ipywidgets.Checkbox',
            return_value=MagicMock()
        )
        self.html_patch = patch(
            'ipywidgets.HTML',
            return_value=MagicMock()
        )
        # Create a mock VBox that properly handles children and layout
        def create_vbox(children=None, **kwargs):
            mock = MagicMock()
            mock.children = children or []
            mock.layout = MagicMock()
            return mock
            
        self.vbox_patch = patch(
            'ipywidgets.VBox',
            side_effect=create_vbox
        )

        # Start all patches
        self.header_patch.start()
        self.status_panel_patch.start()
        self.action_buttons_patch.start()
        self.progress_tracker_patch.start()
        self.log_accordion_patch.start()
        self.module_log_accordion_patch.start()
        self.responsive_container_patch.start()
        self.custom_section_patch.start()
        self.confirmation_area_patch.start()
        self.action_section_patch.start()
        self.checkbox_patch.start()
        self.html_patch.start()
        self.vbox_patch.start()
        
        # Configure the action buttons mock to return a proper widget with children
        mock_buttons = [MagicMock() for _ in range(3)]
        self.mock_action_buttons.children = mock_buttons
        
        # Patch the helper functions
        self.categories_section_patch = patch(
            'smartcash.ui.setup.dependency.components.ui_components._create_categories_section',
            return_value=self.mock_categories_section
        )
        self.mock_create_categories = self.categories_section_patch.start()
        
        self.custom_section_func_patch = patch(
            'smartcash.ui.setup.dependency.components.ui_components._create_custom_package_section',
            return_value=self.mock_custom_section
        )
        self.mock_create_custom_section = self.custom_section_func_patch.start()
        
        # Patch get_default_dependency_config from the correct location
        self.default_config_patch = patch(
            'smartcash.ui.setup.dependency.handlers.defaults.get_default_dependency_config',
            return_value={'categories': []}
        )
        self.mock_get_default_config = self.default_config_patch.start()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop all patches in reverse order
        self.vbox_patch.stop()
        self.html_patch.stop()
        self.checkbox_patch.stop()
        self.action_section_patch.stop()
        self.confirmation_area_patch.stop()
        self.custom_section_patch.stop()
        self.responsive_container_patch.stop()
        self.module_log_accordion_patch.stop()
        self.log_accordion_patch.stop()
        self.progress_tracker_patch.stop()
        self.action_buttons_patch.stop()
        self.status_panel_patch.stop()
        self.header_patch.stop()
        self.logger_patch.stop()
        
        # Stop any other patches that might have been started
        if hasattr(self, 'categories_section_patch'):
            self.categories_section_patch.stop()
        if hasattr(self, 'custom_section_func_patch'):
            self.custom_section_func_patch.stop()
        if hasattr(self, 'default_config_patch'):
            self.default_config_patch.stop()

    def test_create_dependency_main_ui(self):
        """Test creating the main dependency management UI."""
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        
        # Create a test config with required structure
        test_config = {
            'categories': [
                {
                    'name': 'Test Category',
                    'description': 'Test Description',
                    'icon': '📦',
                    'packages': [
                        {
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
                        }
                    ]
                }
            ]
        }
        
        # Reset all mocks to ensure clean state
        self.mock_logger_instance.reset_mock()
        self.mock_status_panel.reset_mock()
        self.mock_action_buttons.reset_mock()
        self.mock_progress_tracker.reset_mock()
        self.mock_log_output.reset_mock()
        self.mock_log_accordion.reset_mock()
        self.mock_entries_container.reset_mock()
        self.mock_action_section.reset_mock()
        self.mock_header.reset_mock()
        
        # Reset any layout mocks
        for widget in [self.mock_header, self.mock_status_panel, self.mock_action_buttons,
                      self.mock_progress_tracker, self.mock_log_output, self.mock_log_accordion,
                      self.mock_entries_container, self.mock_action_section]:
            if hasattr(widget, 'layout'):
                widget.layout.reset_mock()
        
        # Create proper button mocks with the expected attributes
        self.mock_install_btn = MagicMock()
        self.mock_check_updates_btn = MagicMock()
        self.mock_uninstall_btn = MagicMock()
        
        # Create a proper HBox mock for action buttons with children
        self.mock_action_buttons = MagicMock(spec=widgets.HBox)
        self.mock_action_buttons.children = [
            self.mock_install_btn,
            self.mock_check_updates_btn,
            self.mock_uninstall_btn
        ]
        
        # Make the action_buttons_patch return our mock HBox
        self.action_buttons_patch.return_value = self.mock_action_buttons
        
        # Mock the create_action_buttons to return a dictionary with the buttons
        # This matches the actual implementation in the UI components
        self.action_buttons_patch.return_value = {
            'buttons': [
                self.mock_install_btn,
                self.mock_check_updates_btn,
                self.mock_uninstall_btn
            ],
            'count': 3,
            'layout_style': 'flex',
            'primary': self.mock_install_btn,
            'install_btn': self.mock_install_btn,
            'check_updates_btn': self.mock_check_updates_btn,
            'uninstall_btn': self.mock_uninstall_btn
        }
        
        # Call the function with test config
        result = create_dependency_main_ui(test_config)
        
        # Verify the result contains all expected keys
        expected_keys = [
            'ui', 'container', 'header', 'status_panel', 'categories_section',
            'custom_section', 'action_section', 'progress_tracker',
            'log_components', 'logger_bridge', 'install_btn', 'check_updates_btn',
            'uninstall_btn'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result, f"Expected key '{key}' not found in result")
        
        # Verify the logger bridge was created with correct parameters
        self.mock_logger_bridge.assert_called_once()
        call_args = self.mock_logger_bridge.call_args[1]  # Get kwargs
        
        # Verify the logger bridge was called with the correct parameters
        self.assertIn('ui_components', call_args)
        self.assertIn('logger_name', call_args)
        self.assertEqual(call_args.get('logger_name'), 'dependency_ui')
        
        # Verify the UI components were created with correct parameters
        self.status_panel_patch.assert_called_once_with(
            "🚀 Siap mengelola dependencies",
            "info"
        )
        
        # Verify the action buttons were created with correct parameters
        self.action_buttons_patch.assert_called_once_with([
            ("install_btn", "📦 Install", "primary", False),
            ("check_updates_btn", "🔍 Check Updates", "info", False),
            ("uninstall_btn", "🗑️ Uninstall", "danger", False)
        ])
        
        # Verify the progress tracker was created
        self.progress_tracker_patch.assert_called_once_with(
            "Overall Progress", "Current Operation"
        )
        
        # Verify the log accordion was created with correct parameters
        self.log_accordion_patch.assert_called_once_with("Operation Logs", height="300px")
        
        # Verify the responsive container was called with correct children
        self.assertTrue(self.responsive_container_patch.called)
        
        # Verify the action section was created with correct parameters
        self.action_section_patch.assert_called_once()
        
        # Verify the logger bridge instance was stored in the result
        self.assertEqual(result['logger_bridge'], self.mock_logger_instance)
        
        # Verify the action buttons are properly set in the result
        self.assertEqual(result['install_btn'], mock_buttons[0])
        self.assertEqual(result['check_updates_btn'], mock_buttons[1])
        self.assertEqual(result['uninstall_btn'], mock_buttons[2])
        
        # Verify the progress tracker was created with correct parameters
        self.progress_tracker_patch.assert_called_once_with(
            "Overall Progress",
            "Current Operation"
        )
        
        # Verify the log accordion was created with correct parameters
        self.log_accordion_patch.assert_called_once_with("Operation Logs", height="300px")
        
        # Verify the responsive container was called with all components
        self.assertTrue(self.responsive_container_patch.called)
        
        # Verify the action section was created with the correct parameters
        self.action_section_patch.assert_called_once()
        call_args = self.action_section_patch.call_args[1]
        self.assertEqual(call_args['title'], "🚀 Dependency Operations")
        self.assertEqual(call_args['status_label'], "📋 Operation Status:")
        self.assertTrue(call_args['show_status'])

    def test_create_dependency_main_ui_default_config(self):
        """Test creating the main UI with default config."""
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        
        # Mock the logger bridge
        with patch('smartcash.ui.setup.dependency.components.ui_components.UILoggerBridge') as mock_logger_bridge:
            mock_logger_instance = MagicMock()
            mock_logger_bridge.return_value = mock_logger_instance
            
            # Call the function without a config
            result = create_dependency_main_ui()
            
            # Verify the result contains expected keys
            self.assertIn('ui', result)
            self.assertIn('logger_bridge', result)
            
            # Verify logger bridge was created
            mock_logger_bridge.assert_called_once()

    def test_create_categories_section(self):
        """Test creating the package categories section."""
        from smartcash.ui.setup.dependency.components.ui_components import _create_categories_section
        
        # Reset mocks
        self.mock_get_default_config.return_value = {'categories': []}
        self.html_patch.reset_mock()
        self.vbox_patch.reset_mock()
        
        # Test with empty config
        result = _create_categories_section({})
        self.assertIsNotNone(result)
        self.mock_get_default_config.assert_called_once()
        
        # Test with valid config
        test_config = {
            'categories': [
                {
                    'name': 'Test Category',
                    'description': 'Test Description',
                    'icon': '📦',
                    'packages': [
                        {
                            'name': 'test-package', 
                            'description': 'Test package', 
                            'key': 'test-pkg',
                            'pip_name': 'test-package',
                            'required': False
                        }
                    ]
                }
            ]
        }
        
        # Reset mocks
        self.html_patch.reset_mock()
        self.vbox_patch.reset_mock()
        
        # Mock the checkbox creation
        mock_checkbox = MagicMock()
        self.checkbox_patch.return_value = mock_checkbox
        
        # Call the function
        result = _create_categories_section(test_config)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.checkbox_patch.assert_called_once()
        self.html_patch.assert_called()  # For description and other HTML elements
        self.vbox_patch.assert_called()  # For layout

    def test_create_custom_package_section(self):
        """Test creating the custom package section."""
        from smartcash.ui.setup.dependency.components.ui_components import _create_custom_package_section
        
        # Reset mocks
        self.custom_section_patch.reset_mock()
        
        # Test with empty config
        result = _create_custom_package_section({})
        self.assertIsNotNone(result)
        
        # Verify the text input was created with the correct parameters
        self.custom_section_patch.assert_called_with(
            'custom_packages_input',
            'Custom packages (misal: scikit-learn==1.3.0, matplotlib)',
            '',
            multiline=True
        )
        
        # Test with custom packages
        test_config = {'custom_packages': 'numpy>=1.0.0,pandas'}
        result = _create_custom_package_section(test_config)
        self.assertIsNotNone(result)
        
    def test_update_package_status(self):
        """Test updating package status in the UI."""
        from smartcash.ui.setup.dependency.components.ui_components import update_package_status
        
        # Create test UI components
        mock_status = MagicMock()
        mock_logger = MagicMock()
        ui_components = {
            'pkg_test-package': MagicMock(),
            'status_panel': mock_status,
            'logger_bridge': mock_logger
        }
        
        # Test successful update
        update_package_status(ui_components, 'test-package', 'installed')
        
        # Verify the status was updated
        mock_status.update_status.assert_called_once_with(
            '✅ Package test-package installed successfully',
            'success'
        )
        
        # Test package not found
        mock_status.reset_mock()
        update_package_status(ui_components, 'nonexistent', 'installed')
        mock_status.update_status.assert_called_with(
            '⚠️ Package nonexistent not found in UI components',
            'warning'
        )
        ui_components['logger_bridge'].info.assert_called_once()
        
    def test_extract_category_components(self):
        """Test extracting category components."""
        from smartcash.ui.setup.dependency.components.ui_components import _extract_category_components
        
        # Create test widgets
        mock_checkbox = widgets.Checkbox(description='test-package', value=False)
        mock_description = widgets.HTML()
        package_widget = widgets.VBox([mock_checkbox, mock_description])
        mock_category = widgets.VBox([
            widgets.HTML('<div>Test Category</div>'),
            widgets.VBox([package_widget])
        ])
        categories_section = widgets.VBox([mock_category])
        
        # Call the function
        result = _extract_category_components(categories_section)
        
        # Verify the result
        self.assertIn('pkg_test_package', result)
        self.assertEqual(result['pkg_test_package'], mock_checkbox)
        
    def test_extract_custom_components(self):
        """Test extracting custom components."""
        from smartcash.ui.setup.dependency.components.ui_components import _extract_custom_components
        
        # Create test widgets
        mock_input = widgets.Text(placeholder='Enter packages')
        mock_button = widgets.Button(description='Add')
        mock_list = widgets.HTML()
        
        custom_section = widgets.VBox([
            widgets.HTML('<div>Custom Packages</div>'),
            mock_input,
            mock_button,
            mock_list
        ])
        
        # Call the function
        result = _extract_custom_components(custom_section)
        
        # Verify the result
        self.assertIn('custom_packages_input', result)
        self.assertIn('add_custom_btn', result)
        self.assertIn('custom_packages_list', result)

if __name__ == '__main__':
    unittest.main()

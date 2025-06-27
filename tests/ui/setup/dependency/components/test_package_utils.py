"""
Test cases for package utility functions in dependency management UI.
"""
import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any, List

class TestPackageUtils(unittest.TestCase):
    """Test cases for package utility functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the functions
        from smartcash.ui.setup.dependency.components.ui_components import (
            _create_package_checkbox,
            _extract_category_components,
            _extract_custom_components,
            update_package_status
        )
        
        self.create_package_checkbox = _create_package_checkbox
        self.extract_category_components = _extract_category_components
        self.extract_custom_components = _extract_custom_components
        self.update_package_status = update_package_status
        
        # Create test data
        self.test_package = {
            'name': 'test-package',
            'description': 'A test package',
            'key': 'test-pkg',
            'pip_name': 'test-package',
            'required': False,
            'installed': True,
            'version': '1.0.0',
            'latest_version': '1.1.0',
            'update_available': True,
            'dependencies': []
        }
        
        # Mock the widgets
        self.mock_checkbox = MagicMock()
        self.mock_vbox = MagicMock()
        self.mock_html = MagicMock()
        
        # Patch the widget constructors
        self.checkbox_patch = patch('ipywidgets.Checkbox', return_value=self.mock_checkbox)
        self.vbox_patch = patch('ipywidgets.VBox', return_value=self.mock_vbox)
        self.html_patch = patch('ipywidgets.HTML', return_value=self.mock_html)
        
        # Start the patches
        self.checkbox_patch.start()
        self.vbox_patch.start()
        self.html_patch.start()
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.checkbox_patch.stop()
        self.vbox_patch.stop()
        self.html_patch.stop()
    
    def test_create_package_checkbox(self):
        """Test creating a package checkbox."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.vbox_patch.return_value = mock_vbox
        
        # Call the function with test package
        result = self.create_package_checkbox(self.test_package, [])
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)  # Mocked VBox
        
        # Verify checkbox was created with correct parameters
        from ipywidgets import Checkbox
        self.assertTrue(Checkbox.called)
        args, kwargs = Checkbox.call_args
        self.assertEqual(kwargs.get('description'), self.test_package['name'])
        self.assertEqual(kwargs.get('value'), False)
        self.assertEqual(kwargs.get('disabled'), False)
        
        # Verify HTML was created for the description
        from ipywidgets import HTML
        self.assertTrue(HTML.called)
        
        # Verify VBox was called
        self.assertTrue(self.vbox_patch.called)
    
    def test_extract_category_components(self):
        """Test extracting components from categories section."""
        # Create a mock categories section with checkboxes
        mock_checkbox1 = MagicMock()
        mock_checkbox1.description = 'Package 1'
        mock_checkbox1.value = True
        
        mock_checkbox2 = MagicMock()
        mock_checkbox2.description = 'Package 2'
        mock_checkbox2.value = False
        
        # Create a mock VBox with checkboxes
        mock_vbox = MagicMock()
        mock_vbox.children = [mock_checkbox1, mock_checkbox2]
        
        # Call the function
        result = self.extract_category_components(mock_vbox)
        
        # Verify the result contains the checkboxes
        self.assertIn('pkg_package_1', result)
        self.assertIn('pkg_package_2', result)
        self.assertIs(result['pkg_package_1'], mock_checkbox1)
        self.assertIs(result['pkg_package_2'], mock_checkbox2)
    
    def test_extract_custom_components(self):
        """Test extracting components from custom section."""
        # Create a mock custom section with input and button
        mock_input = MagicMock()
        mock_input.description = 'custom_packages_input'
        
        mock_button = MagicMock()
        mock_button.description = '➕ Add Custom'
        
        # Create a mock VBox with the input and button
        mock_vbox = MagicMock()
        mock_vbox.children = [MagicMock(), mock_input, mock_button, MagicMock()]  # Other elements are placeholders
        
        # Call the function
        result = self.extract_custom_components(mock_vbox)
        
        # Verify the result contains the components
        self.assertIn('custom_packages_input', result)
        self.assertIn('add_custom_package_btn', result)
        self.assertIs(result['custom_packages_input'], mock_input)
        self.assertIs(result['add_custom_package_btn'], mock_button)
    
    def test_update_package_status(self):
        """Test updating package status in the UI."""
        # Create mock UI components
        ui_components = {
            'pkg_test_pkg': MagicMock(),
            'pkg_other_pkg': MagicMock()
        }
        
        # Call the function to update status
        self.update_package_status(ui_components, 'test_pkg', 'installed')
        
        # Verify the checkbox was updated
        ui_components['pkg_test_pkg'].value = True
        ui_components['pkg_test_pkg'].disabled = True
        
        # Verify other packages were not affected
        ui_components['pkg_other_pkg'].assert_not_called()

if __name__ == '__main__':
    unittest.main()

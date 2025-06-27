"""
Test cases for the custom package section component in dependency management UI.
"""
import unittest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch, ANY
from typing import Dict, Any

class TestCustomPackageSection(unittest.TestCase):
    """Test cases for the custom package section component."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the function
        from smartcash.ui.setup.dependency.components.ui_components import _create_custom_package_section
        self.create_custom_section = _create_custom_package_section
        
        # Create test data
        self.test_config = {
            'custom_packages': 'requests==2.25.1, pandas>=1.2.0'
        }
        
        # Mock the widgets
        self.mock_button = MagicMock()
        self.mock_text_input = MagicMock()
        self.mock_vbox = MagicMock()
        self.mock_html = MagicMock()
        
        # Patch the widget constructors
        self.button_patch = patch('ipywidgets.Button', return_value=self.mock_button)
        self.text_input_patch = patch('smartcash.ui.components.create_text_input', 
                                     return_value=self.mock_text_input)
        self.vbox_patch = patch('ipywidgets.VBox', return_value=self.mock_vbox)
        self.html_patch = patch('ipywidgets.HTML', return_value=self.mock_html)
        
        # Start the patches
        self.button_patch.start()
        self.text_input_patch.start()
        self.vbox_patch.start()
        self.html_patch.start()
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.button_patch.stop()
        self.text_input_patch.stop()
        self.vbox_patch.stop()
        self.html_patch.stop()
    
    def test_create_custom_section_with_config(self):
        """Test creating custom section with provided config."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.vbox_patch.return_value = mock_vbox
        
        # Call the function with test config
        result = self.create_custom_section(self.test_config)
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)  # Mocked VBox
        
        # Verify text input was created with the correct parameters
        from smartcash.ui.components import create_text_input
        self.assertTrue(create_text_input.called)
        args, kwargs = create_text_input.call_args
        self.assertEqual(kwargs.get('value'), self.test_config['custom_packages'])
        
        # Verify button was created with the correct parameters
        from ipywidgets import Button
        self.assertTrue(Button.called)
        button_args, button_kwargs = Button.call_args
        self.assertEqual(button_kwargs.get('description'), '➕ Add Custom')
        self.assertEqual(button_kwargs.get('button_style'), 'info')
        
        # Verify VBox was called with correct arguments
        self.assertTrue(self.vbox_patch.called)
    
    def test_create_custom_section_without_config(self):
        """Test creating custom section without config."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.vbox_patch.return_value = mock_vbox
        
        # Call the function without config
        result = self.create_custom_section({})
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)  # Mocked VBox
        
        # Verify text input was created with empty value
        from smartcash.ui.components import create_text_input
        self.assertTrue(create_text_input.called)
        args, kwargs = create_text_input.call_args
        self.assertEqual(kwargs.get('value'), '')
        
        # Verify VBox was called
        self.assertTrue(self.vbox_patch.called)

if __name__ == '__main__':
    unittest.main()

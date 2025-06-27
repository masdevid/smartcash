"""
Test cases for the custom package section component in dependency management UI.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY

class TestCustomPackageSection(unittest.TestCase):
    """Test cases for the custom package section component."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_config = {
            'custom_packages': 'numpy>=1.0.0, pandas>=1.0.0'
        }
        
        # Patch the widget constructors
        self.text_input_patch = patch('smartcash.ui.components.create_text_input')
        self.button_patch = patch('ipywidgets.Button')
        self.vbox_patch = patch('ipywidgets.VBox')
        self.html_patch = patch('ipywidgets.HTML')
        
        # Start the patches and get the mock objects
        self.mock_text_input = self.text_input_patch.start()
        self.mock_button = self.button_patch.start()
        self.mock_vbox = self.vbox_patch.start()
        self.mock_html = self.html_patch.start()
        
        # Import the function after patching
        from smartcash.ui.setup.dependency.components.ui_components import _create_custom_package_section
        self.create_custom_section = _create_custom_package_section
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.text_input_patch.stop()
        self.button_patch.stop()
        self.vbox_patch.stop()
        self.html_patch.stop()
    
    def test_create_custom_section_with_config(self):
        """Test creating custom section with provided config."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.mock_vbox.return_value = mock_vbox
        
        # Setup mock text input return value
        mock_text_input = MagicMock()
        self.mock_text_input.return_value = mock_text_input
        
        # Setup mock button return value
        mock_button = MagicMock()
        self.mock_button.return_value = mock_button
        
        # Setup mock HTML return value
        mock_html = MagicMock()
        self.mock_html.return_value = mock_html
        
        # Call the function with test config
        result = self.create_custom_section(self.test_config)
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)
        
        # Verify text input was created with the correct parameters
        self.assertTrue(self.mock_text_input.called)
        args, kwargs = self.mock_text_input.call_args
        self.assertEqual(kwargs.get('value'), self.test_config['custom_packages'])
        self.assertEqual(kwargs.get('description'), 'Custom packages (misal: scikit-learn==1.3.0, matplotlib)')
        self.assertTrue(kwargs.get('multiline', False))
        
        # Verify button was created with the correct parameters
        self.assertTrue(self.mock_button.called)
        button_args, button_kwargs = self.mock_button.call_args
        self.assertEqual(button_kwargs.get('description'), '➕ Add Custom')
        self.assertEqual(button_kwargs.get('button_style'), 'info')
        
        # Verify HTML was created for the header
        self.assertTrue(self.mock_html.called)
        
        # Verify VBox was called with correct number of children
        self.assertTrue(self.mock_vbox.called)
        vbox_args, vbox_kwargs = self.mock_vbox.call_args
        self.assertEqual(len(vbox_args[0]), 4)  # header, input, button, html
        
        # Verify the result is the mock VBox
        self.assertEqual(result, mock_vbox)
    
    def test_create_custom_section_without_config(self):
        """Test creating custom section without config."""
        # Setup mock return values
        mock_vbox = MagicMock()
        self.mock_vbox.return_value = mock_vbox
        
        # Setup mock text input return value
        mock_text_input = MagicMock()
        self.mock_text_input.return_value = mock_text_input
        
        # Call the function without config
        result = self.create_custom_section({})
        
        # Verify the result is a VBox
        self.assertIsInstance(result, MagicMock)
        
        # Verify text input was created with empty value
        self.assertTrue(self.mock_text_input.called)
        args, kwargs = self.mock_text_input.call_args
        self.assertEqual(kwargs.get('value'), '')
        
        # Verify VBox was called
        self.assertTrue(self.mock_vbox.called)
        
        # Verify the result is the mock VBox
        self.assertEqual(result, mock_vbox)
        
        # Verify VBox children
        vbox_args, vbox_kwargs = self.mock_vbox.call_args
        self.assertEqual(len(vbox_args[0]), 4)  # header, input, button, html

if __name__ == '__main__':
    unittest.main()

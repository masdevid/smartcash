"""
Test cases for the custom package section component in dependency management UI.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY
import ipywidgets as widgets

class TestCustomPackageSection(unittest.TestCase):
    """Test cases for the custom package section component."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_config = {
            'custom_packages': 'numpy>=1.0.0, pandas>=1.0.0'
        }
        
        # Create a mock for the layout
        self.mock_layout = MagicMock()
        
        # Create patches for all the widget constructors
        self.patchers = [
            patch('smartcash.ui.components.create_text_input'),
            patch('ipywidgets.Button'),
            patch('ipywidgets.VBox'),
            patch('ipywidgets.HTML'),
            patch('ipywidgets.Layout', return_value=self.mock_layout)
        ]
        
        # Start all patches and store the mock objects
        for patcher in self.patchers:
            patcher.start()
        
        # Import the function after patching
        from smartcash.ui.setup.dependency.components.ui_components import _create_custom_package_section
        self.create_custom_section = _create_custom_package_section
        
        # Get the mock objects
        from smartcash.ui import components as ui_components
        import ipywidgets as widgets
        
        self.mock_text_input = ui_components.create_text_input
        self.mock_button = widgets.Button
        self.mock_vbox = widgets.VBox
        self.mock_html = widgets.HTML
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop all patches
        for patcher in self.patchers:
            patcher.stop()
    
    def test_create_custom_section_with_config(self):
        """Test creating custom section with provided config."""
        # Setup return values for mocks
        mock_vbox_instance = MagicMock()
        self.mock_vbox.return_value = mock_vbox_instance
        
        # Call the function with test config
        result = self.create_custom_section(self.test_config)
        
        # Verify the result is the VBox instance
        self.assertIs(result, mock_vbox_instance)
        
        # Verify widget constructors were called with correct parameters
        # Check ipywidgets.HTML was called for the header
        expected_header = """
    <div style='margin: 20px 0 10px 0; padding: 10px; background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 8px;'>
        <h4 style='margin: 0; color: #1976d2;'>⚙️ Custom Packages</h4>
        <p style='margin: 5px 0 0 0; color: #424242; font-size: 0.9em;'>Tambahkan package custom (pisahkan dengan koma)</p>
    </div>
    """.strip()
        
        # Check for the header HTML
        self.mock_html.assert_any_call(expected_header)
        
        # Check for the custom list HTML
        expected_list = "<div style='margin-top: 10px;'></div>"
        self.mock_html.assert_any_call(value=expected_list)
        
        # Check create_text_input was called with correct parameters
        self.mock_text_input.assert_called_once_with(
            "custom_packages_input",
            "Custom packages (misal: scikit-learn==1.3.0, matplotlib)",
            'numpy>=1.0.0, pandas>=1.0.0',
            multiline=True
        )
        
        # Check Button was created with correct parameters
        self.mock_button.assert_called_once_with(
            description='➕ Add Custom',
            button_style='info',
            layout=self.mock_layout
        )
        
        # Check VBox was created with 4 children (header, input, button, empty div)
        self.mock_vbox.assert_called_once()
        vbox_args, _ = self.mock_vbox.call_args
        self.assertEqual(len(vbox_args[0]), 4)
        
        # Verify layout was created with correct width
        widgets.Layout.assert_called_once_with(width='150px')
    
    def test_create_custom_section_without_config(self):
        """Test creating custom section without config."""
        # Setup return values for mocks
        mock_vbox_instance = MagicMock()
        self.mock_vbox.return_value = mock_vbox_instance
        
        # Call the function without config
        result = self.create_custom_section({})
        
        # Verify the result is the VBox instance
        self.assertIs(result, mock_vbox_instance)
        
        # Verify text input was created with empty value
        self.mock_text_input.assert_called_once_with(
            "custom_packages_input",
            "Custom packages (misal: scikit-learn==1.3.0, matplotlib)",
            '',
            multiline=True
        )
        
        # Verify VBox was created with 4 children
        self.mock_vbox.assert_called_once()
        vbox_args, _ = self.mock_vbox.call_args
        self.assertEqual(len(vbox_args[0]), 4)

if __name__ == '__main__':
    unittest.main()

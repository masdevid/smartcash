"""
Test cases for the custom package section component in dependency management UI.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY, call
import ipywidgets as widgets

class TestCustomPackageSection(unittest.TestCase):
    """Test cases for the custom package section component."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test data
        self.test_config = {
            'custom_packages': 'numpy>=1.0.0, pandas>=1.0.0'
        }
        
        # Create mock widget instances
        self.mock_text_input = MagicMock()
        self.mock_button = MagicMock()
        self.mock_vbox = MagicMock()
        self.mock_html = MagicMock()
        self.mock_layout = MagicMock()
        
        # Configure the VBox to return our mock instance when called
        self.mock_vbox.return_value = self.mock_vbox
        
        # Set up patches with side effects to return our mock instances
        self.patchers = [
            patch('smartcash.ui.components.create_text_input', 
                 return_value=self.mock_text_input),
            patch('ipywidgets.Button', 
                 return_value=self.mock_button),
            patch('ipywidgets.VBox', 
                 return_value=self.mock_vbox),
            patch('ipywidgets.HTML', 
                 side_effect=[self.mock_html, self.mock_html]),  # Called twice
            patch('ipywidgets.Layout', 
                 return_value=self.mock_layout)
        ]
        
        # Start all patches
        for patcher in self.patchers:
            patcher.start()
        
        # Import the function after patching
        from smartcash.ui.setup.dependency.components import ui_components
        self.create_custom_section = ui_components._create_custom_package_section
        
        # Store the actual modules for assertions
        self.ui_components = ui_components
        import ipywidgets as widgets
        self.widgets = widgets
    
    def test_create_custom_section_with_config(self):
        """Test creating custom section with provided config."""
        # Call the function with test config
        result = self.create_custom_section(self.test_config)
        
        # Verify the result is the mock VBox instance
        self.assertIs(result, self.mock_vbox, 
                    f"Expected result to be {self.mock_vbox}, but got {result}")
        
        # Verify widget constructors were called with correct parameters
        # Check ipywidgets.HTML was called for the header and list
        expected_header = """
    <div style='margin: 20px 0 10px 0; padding: 10px; background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 8px;'>
        <h4 style='margin: 0; color: #1976d2;'>⚙️ Custom Packages</h4>
        <p style='margin: 5px 0 0 0; color: #424242; font-size: 0.9em;'>Tambahkan package custom (pisahkan dengan koma)</p>
    </div>
    """.strip()
        
        # Check HTML widget was called twice (for header and list)
        self.assertEqual(self.widgets.HTML.call_count, 2, 
                        f"Expected 2 calls to HTML, got {self.widgets.HTML.call_count}")
        
        # Check the header HTML call
        self.widgets.HTML.assert_any_call(expected_header)
        
        # Check the custom list HTML call
        self.widgets.HTML.assert_any_call(value="<div style='margin-top: 10px;'></div>")
        
        # Check create_text_input was called with correct parameters
        self.mock_text_input.assert_called_once_with(
            "custom_packages_input",
            "Custom packages (misal: scikit-learn==1.3.0, matplotlib)",
            'numpy>=1.0.0, pandas>=1.0.0',
            multiline=True
        )
        
        # Check Button was created with correct parameters
        self.widgets.Button.assert_called_once_with(
            description='➕ Add Custom',
            button_style='info',
            layout=self.mock_layout
        )
        
        # Check VBox was created with 4 children (header, input, button, empty div)
        self.widgets.VBox.assert_called_once()
        vbox_args, _ = self.widgets.VBox.call_args
        self.assertEqual(len(vbox_args[0]), 4, 
                        f"Expected VBox to be created with 4 children, got {len(vbox_args[0])}")
        
        # Verify layout was created with correct width
        self.widgets.Layout.assert_called_once_with(width='150px')
        
        # Verify the VBox children are in the correct order
        children = vbox_args[0]
        self.assertIs(children[0], self.mock_html, "First child should be the header HTML")
        self.assertIs(children[1], self.mock_text_input, "Second child should be the text input")
        self.assertIs(children[2], self.mock_button, "Third child should be the button")
        self.assertIs(children[3], self.mock_html, "Fourth child should be the custom list HTML")
    
    def test_create_custom_section_without_config(self):
        """Test creating custom section without config."""
        # Reset mocks to clear any calls from setup
        self.widgets.HTML.reset_mock()
        self.mock_text_input.reset_mock()
        self.widgets.Button.reset_mock()
        self.widgets.VBox.reset_mock()
        self.widgets.Layout.reset_mock()
        
        # Call the function without config
        result = self.create_custom_section({})
        
        # Verify the result is the mock VBox instance
        self.assertIs(result, self.mock_vbox, 
                    f"Expected result to be {self.mock_vbox}, but got {result}")
        
        # Verify widget constructors were called with correct parameters
        # Check ipywidgets.HTML was called for the header and list
        expected_header = """
    <div style='margin: 20px 0 10px 0; padding: 10px; background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 8px;'>
        <h4 style='margin: 0; color: #1976d2;'>⚙️ Custom Packages</h4>
        <p style='margin: 5px 0 0 0; color: #424242; font-size: 0.9em;'>Tambahkan package custom (pisahkan dengan koma)</p>
    </div>
    """.strip()
        
        # Check HTML widget was called twice (for header and list)
        self.assertEqual(self.widgets.HTML.call_count, 2, 
                        f"Expected 2 calls to HTML, got {self.widgets.HTML.call_count}")
        
        # Check the header HTML call
        self.widgets.HTML.assert_any_call(expected_header)
        
        # Check the custom list HTML call
        self.widgets.HTML.assert_any_call(value="<div style='margin-top: 10px;'></div>")
        
        # Verify text input was created with empty value
        self.mock_text_input.assert_called_once_with(
            "custom_packages_input",
            "Custom packages (misal: scikit-learn==1.3.0, matplotlib)",
            '',
            multiline=True
        )
        
        # Check Button was created with correct parameters
        self.widgets.Button.assert_called_once_with(
            description='➕ Add Custom',
            button_style='info',
            layout=self.mock_layout
        )
        
        # Check VBox was created with 4 children (header, input, button, empty div)
        self.widgets.VBox.assert_called_once()
        vbox_args, _ = self.widgets.VBox.call_args
        self.assertEqual(len(vbox_args[0]), 4, 
                        f"Expected VBox to be created with 4 children, got {len(vbox_args[0])}")
        
        # Verify layout was created with correct width
        self.widgets.Layout.assert_called_once_with(width='150px')
        
        # Verify the VBox children are in the correct order
        children = vbox_args[0]
        self.assertIs(children[0], self.mock_html, "First child should be the header HTML")
        self.assertIs(children[1], self.mock_text_input, "Second child should be the text input")
        self.assertIs(children[2], self.mock_button, "Third child should be the button")
        self.assertIs(children[3], self.mock_html, "Fourth child should be the custom list HTML")
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop all patches
        for patcher in self.patchers:
            patcher.stop()

if __name__ == '__main__':
    unittest.main()
